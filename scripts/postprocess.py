"""Source-language subtitle shaping; text models never supply timestamps."""
from dataclasses import replace
from difflib import SequenceMatcher
import re
import unicodedata

UNIT = re.compile(r'^\s*\[(\d+)(?:\s*[-–~]\s*(\d+))?\]\s*(\S.*)$')
SENTENCE = re.compile(r'[。！？!?…]|\.(?=\s|$)')
CJK = re.compile(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]')

REFINE_PROMPT = '''整理原语种自动听写字幕。把被机械截断的同一句连续碎行合并，补必要标点，只做有充分上下文依据的轻微同音纠正。保留内容、说话口吻、有意义的重复、疑问和否定；不同说话人不要合并。禁止翻译、改写、补写漏掉的台词、拆出新条目或输出时间。每一条编号按序覆盖一次。每行输出 [编号] 原文，合并时输出 [首号-尾号] 原文。上一块的参考文字不用输出。只输出这些行。'''
LYRICS_PROMPT = '''找出这些带编号原语种字幕中唱出来的内容，包括片头片尾歌、背景插曲和人物唱歌或有可辨词句的哼唱。不标正常对话、独白、朗读和仅有文学措辞的台词；不确定时不标。每个连续歌词范围输出一行：[起始编号-结束编号] 起始行开头连续几个原字 | 结束行结尾连续几个原字。单条可写 [编号]。摘录必须从指定行首尾照抄；不要改写，不要加音符，不要输出时间。没有歌词只输出「无」。'''


def join(a, b):
    if not a:
        return b
    if CJK.search(a[-1]) and CJK.search(b[0]):
        return a+b
    if b[0] in '。？！、，；：,.!?;:…' or a.endswith((' ', '\n')):
        return a+b
    return a+' '+b


def split_text(text, width):
    out = []
    while len(text) > width:
        target = min(width,len(text)-1)
        possible = [m.end() for m in SENTENCE.finditer(text[:target+1]) if m.end()>=max(1,target//2)]
        if not possible:
            possible = [i+1 for i,c in enumerate(text[:target])
                        if i>=target//2 and (c.isspace() or c in '，、,；;：:')]
        if possible:
            cut = possible[-1]
        elif CJK.search(text[:target]):
            cut=target
        else:
            spaces=[i for i,c in enumerate(text) if c.isspace() and i>target]
            if not spaces:
                break
            cut=spaces[0]
        out.append(text[:cut].strip())
        text=text[cut:].strip()
    return [piece for piece in (*out,text) if piece]


def segment(cues, max_chars=42, max_duration=6000):
    result=[]
    for cue in cues:
        span=cue.end-cue.start
        width=max_chars
        if span>max_duration and len(cue.text)>24:
            width=min(width,max(16,round(len(cue.text)*max_duration/span)))
        pieces=split_text(cue.text,width*2 if width==max_chars and span<=max_duration else width)
        if len(pieces)<=1 or span < 500*len(pieces):
            result.append(cue)
            continue
        weights=[len(p) for p in pieces]
        total=sum(weights)
        elapsed=0
        for piece,size in zip(pieces,weights):
            start=cue.start+round(span*elapsed/total)
            elapsed+=size
            end=cue.start+round(span*elapsed/total)
            result.append(replace(cue,start=start,end=end,text=piece))
    return result


def display_wrap(text,width):
    """Hard breaks only for CJK; retain existing cue timing and wording."""
    if '\n' in text or len(text)<=width:
        return text
    middle=len(text)//2
    low=max(1,len(text)-width)
    high=min(width,len(text)-1)
    choices=[i for i in range(low,high+1)
             if text[i].isspace() or text[i-1] in '。？！，、；：,.!?;:']
    if choices:
        cut=min(choices,key=lambda i:abs(i-middle))
    elif CJK.search(text):
        cut=middle
    else:
        return text
    return text[:cut].rstrip()+'\n'+text[cut:].lstrip()


def tokens(text):
    norm=unicodedata.normalize('NFKC',text).casefold()
    return re.findall(r'[a-z\d]+|[^\W\d_a-z]',norm,re.UNICODE)


def faithful(new, old):
    a,b=tokens(old),tokens(new)
    if not a or not b:
        return False
    if a==b:
        return True
    nums=lambda x: [t for t in x if t.isnumeric()]
    if nums(a)!=nums(b):
        return False
    same=sum(m.size for m in SequenceMatcher(a=a,b=b,autojunk=False).get_matching_blocks())
    return same/len(a)>=.68 and len(b)-same<=max(2,len(b)*.25)


def batches(cues,max_estimated):
    """Split a request independently from subtitle cue grouping."""
    start=0
    used=0
    for i,cue in enumerate(cues):
        cost=8+len(cue.text.encode('utf-8'))//2
        if used and used+cost>max_estimated:
            yield start,i
            start=i
            used=0
        used+=cost
    if start<len(cues):
        yield start,len(cues)


def numbered(cues,start,stop):
    return '\n'.join(f'[{i+1}] '+re.sub(r'\s+',' ',cues[i].text).strip() for i in range(start,stop))


def units(reply,start,stop):
    result=[]
    expected=start+1
    for line in reply.splitlines():
        if not line.strip() or line.lstrip().startswith('```'):
            continue
        m=UNIT.fullmatch(line)
        if not m:
            raise ValueError('原文整理返回了无法解析的行')
        a,b,text=m.groups()
        a,b=int(a),int(b or a)
        if a!=expected or b<a or b>stop:
            raise ValueError('原文行号有缺漏、重复或越界')
        result.append((a-1,b,text))
        expected=b+1
    if expected!=stop+1:
        raise ValueError('原文行号覆盖不完整')
    return result


def refine(cues, ask, *, max_chars=42, max_duration=6000):
    result=[]
    info={'batches':0,'fallback_batches':0,'rejected_units':0,'merged':0,'text_edits':0,'errors':[]}
    failures=0
    gaps=sorted(b.start-a.end for a,b in zip(cues,cues[1:]) if b.start>a.end)
    pause=min(2500,max(1200,gaps[int(len(gaps)*.4)])) if gaps else 1200
    for start,stop in batches(cues,1500):
        if failures>=3:
            result.extend(cues[start:stop])
            info['fallback_batches']+=1
            continue
        info['batches']+=1
        context='参考上文，不输出：\n'+'\n'.join(c.text for c in result[-2:])+'\n' if result else ''
        try:
            groups=units(ask(REFINE_PROMPT,context+numbered(cues,start,stop)),start,stop)
        except (RuntimeError, ValueError, OSError) as exc:
            info['fallback_batches']+=1
            info['errors'].append(str(exc)[:130])
            failures+=1
            result.extend(cues[start:stop])
            continue
        failures=0
        for a,b,text in groups:
            src=cues[a:b]
            original=''
            for c in src:
                original=join(original,c.text)
            allowed=faithful(text,original)
            if len(src)>1:
                char_limit=max_chars*2
                allowed=allowed and len(text)<=char_limit and src[-1].end-src[0].start<=max(7000,max_duration)
                allowed=allowed and all(y.start-x.end<=pause for x,y in zip(src,src[1:]))
            if allowed:
                result.append(replace(src[0],end=src[-1].end,text=text))
                info['merged']+=len(src)-1
                info['text_edits']+=text!=original
            else:
                info['rejected_units']+=1
                result.extend(src)
    return result,info


def normalized(text):
    return ''.join(tokens(text))


def mark_lyrics(cues, ask):
    selected=set()
    info={'batches':0,'fallback_batches':0,'rejected_ranges':0,'ranges':[],'errors':[]}
    failures=0
    for start,stop in batches(cues,10000):
        if failures>=2:
            info['fallback_batches']+=1
            continue
        info['batches']+=1
        try:
            reply=ask(LYRICS_PROMPT,numbered(cues,start,stop))
            failures=0
        except (RuntimeError, ValueError, OSError) as exc:
            info['fallback_batches']+=1
            info['errors'].append(str(exc)[:130])
            failures+=1
            continue
        if reply.strip().rstrip('。').casefold() in ('无','none','no lyrics'):
            continue
        last=start
        for line in reply.splitlines():
            if not line.strip() or line.lstrip().startswith('```'):
                continue
            match=UNIT.fullmatch(line)
            if not match:
                info['rejected_ranges']+=1
                continue
            a,b,snips=match.groups()
            a,b=int(a),int(b or a)
            h,sep,t=snips.partition('|')
            h,t=normalized(h),normalized(t)
            if (not sep or not h or not t or not last<a<=b<=stop or
                    b-a+1>max(60,int(len(cues)*.20)) or
                    not normalized(cues[a-1].text).startswith(h) or
                    not normalized(cues[b-1].text).endswith(t)):
                info['rejected_ranges']+=1
                continue
            last=b
            selected.update(range(a-1,b))
            info['ranges'].append([a,b])
    out=[replace(c,text=f'♪ {c.text} ♪') if i in selected and not (
         c.text.startswith('♪ ') and c.text.endswith(' ♪')) else c for i,c in enumerate(cues)]
    info['marked']=len(selected)
    return out,info
