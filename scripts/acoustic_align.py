"""Full-film trial: source-language refinement, offline Japanese CTC, then lyrics."""
from collections import defaultdict, Counter
from dataclasses import replace
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import onnxruntime as ort
import gemini_transcribe as g
import postprocess as pp
from ctc_core import forced, prepare, emissions


def plan_parts(text, width):
    # Only propose sensible sentence boundaries. Formatting inside a cue happens later.
    pieces = [x.strip() for x in re.split(r'(?<=[。！？!?])(?=[^」』）】!?！？])', text) if x.strip()]
    if len(text) <= width or len(pieces) < 2 or any(len(x) < 6 for x in pieces):
        return [text]
    return pieces


def align_japanese(media, cues, film_end, model, width, progress=True):
    vocab=json.loads((model/'vocab.json').read_text(encoding='utf-8'))
    config=json.loads((model/'config.json').read_text(encoding='utf-8'))
    stride=int(np.prod(config['conv_stride']))/16
    blank=config.get('pad_token_id',vocab.get('<pad>'))
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model/'model_int8.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    groups=defaultdict(list)
    pad=3200
    for i,c in enumerate(cues):
        groups[(c.start+c.end)//44000].append((i,c,plan_parts(c.text,width)))
    proposals, confidence = {}, {}
    counts=Counter()
    wav = Path(media)
    for n, group in enumerate(groups.values(),1):
        a=max(0,min(c.start for _,c,_ in group)-pad-400)
        b=min(film_end,max(c.end for _,c,_ in group)+pad+400)
        logp=emissions(wav,a,b,session)
        for i,c,parts in group:
            tokens=[prepare(p,vocab) for p in parts]
            labels=[t for block,_ in tokens for t in block]
            if len(labels)<4 or min(r for _,r in tokens)<.86 or any(not seq for seq,_ in tokens):
                counts['unsupported_or_short']+=1
                continue
            left=max(0,int((c.start-pad-a)/stride))
            right=min(len(logp),int(np.ceil((c.end+pad-a)/stride)))
            try:
                obs=forced(logp[left:right],labels,blank,(c.start-a)/stride-left,(c.end-a)/stride-left)
            except ValueError:
                counts['no_path']+=1
                continue
            probs=np.array([v[2] for v in obs])
            score=float(probs.mean())
            if score < (.73 if len(labels)<7 else .61):
                counts['weak_evidence']+=1
                continue
            def bounds(first,last):
                return (max(0,round(a+(left+obs[first][0])*stride-75)),
                        min(film_end,round(a+(left+obs[last-1][1]+1)*stride+145)))
            def edge(k, tail=False):
                pair=probs[max(0,k-2):k] if tail else probs[k:k+2]
                return float(pair.mean())>=.48
            aa,bb=bounds(0,len(obs))
            aa=aa if edge(0) and 135<=abs(aa-c.start)<=2800 else c.start
            bb=bb if edge(len(obs),True) and 135<=abs(bb-c.end)<=2800 else c.end
            if bb-aa<min(500,c.end-c.start):
                counts['unsafe_boundary']+=1
                continue
            candidate=[replace(c,start=aa,end=bb)]
            if len(parts)>1 and score>=.70:
                splits=[]
                k=0
                good=True
                for piece,(seq,_) in zip(parts,tokens):
                    j=k+len(seq)
                    x,y=bounds(k,j)
                    if not edge(k) or not edge(j,True) or y-x<650:
                        good=False
                        break
                    splits.append(replace(c,start=x,end=y,text=piece))
                    k=j
                if good:
                    splits[0]=replace(splits[0],start=aa)
                    splits[-1]=replace(splits[-1],end=bb)
                    for j in range(len(splits)-1):
                        before,after=splits[j:j+2]
                        if before.end>after.start:
                            cut=(before.end+after.start)//2
                            splits[j]=replace(before,end=cut)
                            splits[j+1]=replace(after,start=cut)
                    if all(x.end-x.start>=600 for x in splits):
                        candidate=splits
            if candidate==[c]:
                counts['near_or_weak_boundary']+=1
                continue
            proposals[i]=candidate
            confidence[i]=score
        if progress and (n%20==0 or n==len(groups)):
            print(f'本地声学分析 {n}/{len(groups)} 段',flush=True)

    # Cap overlaps caused solely by small display padding, otherwise keep originals.
    out=[[c] for c in cues]
    for i, candidate in proposals.items():
        out[i]=candidate
    while True:
        conflict=next((i for i in range(len(out)-1) if out[i][-1].end>out[i+1][0].start),None)
        if conflict is None:
            break
        i=conflict
        eligible=[j for j in (i,i+1) if j in proposals]
        assert eligible
        l,r=out[i][-1],out[i+1][0]
        if l.end-r.start<=160:
            # Trim display padding only on a candidate; never modify untouched cues.
            cut=(l.end+r.start)//2 if len(eligible)==2 else (r.start if i in proposals else l.end)
            if cut-l.start>=min(500,l.end-l.start) and r.end-cut>=min(500,r.end-r.start):
                if i in proposals:
                    out[i][-1]=replace(l,end=cut)
                if i+1 in proposals:
                    out[i+1][0]=replace(r,start=cut)
                continue
        j=min(eligible,key=lambda k:confidence[k])
        out[j]=[cues[j]]
        del proposals[j]
        counts['avoided_neighbor_overlap']+=1

    flat=[c for group in out for c in group]
    norm=lambda x: re.sub(r'\s+','',x)
    assert all(norm(c.text)==norm(''.join(s.text for s in group)) for c,group in zip(cues,out))
    assert all(x.end<=y.start for x,y in zip(flat,flat[1:]))
    assert all(0<=x.start<x.end<=film_end for x in flat)
    counts['changed']=sum(group!=[c] for group,c in zip(out,cues))
    counts['split']=sum(len(group)>1 for group in out)
    counts['unchanged']=len(cues)-counts['changed']
    shifts=[]
    for i,(old,items) in enumerate(zip(cues,out),1):
        if [old]==items:
            continue
        sd,ed=items[0].start-old.start,items[-1].end-old.end
        shifts.append(dict(refined_cue=i,original=[g.clock(old.start),g.clock(old.end)],
                           revised=[[g.clock(c.start),g.clock(c.end)] for c in items],
                           start_delta_ms=sd,end_delta_ms=ed))
    return flat,dict(counts,changes=shifts)



def align_spanish(media, cues, film_end, model, width, progress=True):
    vocab = json.loads((model/'vocab.json').read_text(encoding='utf-8-sig'))
    config = json.loads((model/'config.json').read_text(encoding='utf-8-sig'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model/'model_int8.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    stride = int(np.prod(config['conv_stride']))/16
    assert all(c.end>c.start for c in cues)
    assert all(x.end<=y.start for x,y in zip(cues,cues[1:]))
    groups = defaultdict(list)
    for i,c in enumerate(cues):
        groups[(c.start+c.end)//48000].append((i,c))
    choices = {}
    reasons = {}
    limit = film_end
    pad = 3500
    wav = Path(media)
    for n,group in enumerate(groups.values(),1):
        a=max(0,min(c.start for _,c in group)-pad-600)
        b=min(limit,max(c.end for _,c in group)+pad+600)
        logp=emissions(wav,a,b,session)
        assert logp.shape[-1]==len(vocab)
        for i,c in group:
            original=unicodedata.normalize('NFKC',re.sub(r'<[^>]*>','',c.text)).casefold()
            words=re.findall(r'[^\W\d_]+',original)
            if not words or re.search(r'\d',original) or any(k not in vocab for w in words for k in w):
                reasons[i]='文本含数字或字表外字符，保留'
                continue
            labels=[vocab[k] for k in '|'.join(words)]
            if sum(map(len,words))<4:
                reasons[i]='字幕过短，保留'
                continue
            left=max(0,int((c.start-pad-a)/stride))
            right=min(len(logp),int(np.ceil((c.end+pad-a)/stride)))
            try:
                obs=forced(logp[left:right],labels,config.get('pad_token_id',vocab.get('<pad>')),
                           (c.start-a)/stride-left,(c.end-a)/stride-left)
            except ValueError:
                reasons[i]='未找到对齐路径，保留'
                continue
            groupscore=[]
            p=0
            for w in words:
                groupscore.append(float(np.mean([row[2] for row in obs[p:p+len(w)]])))
                p+=len(w)+1
            score=float(np.mean(groupscore))
            if score < (.77 if len(words)<=2 else .62):
                reasons[i]='声学支持偏弱，保留'
                continue
            start=max(0,round(a+(left+obs[0][0])*stride-80))
            end=min(limit,round(a+(left+obs[-1][1]+1)*stride+160))
            if end-start < min(500,c.end-c.start):
                reasons[i]='对齐区间过短，保留'
                continue
            choices[i] = (
                start if groupscore[0]>=.48 and 140<=abs(start-c.start)<=3000 else c.start,
                end if groupscore[-1]>=.48 and 140<=abs(end-c.end)<=3000 else c.end,
                score,
            )
        if n%5==0 or n==len(groups):
            print(f'已处理 {n}/{len(groups)} 个声学分析段',flush=True)

    # Keep neighbouring cues ordered. First apply moves freeing space, then extend
    # into available space, strongest proposals first. Do not slide another cue.
    retimed=[[c.start,c.end] for c in cues]
    for reverse in [False,True]:
        for i,(start,end,score) in sorted(choices.items(),key=lambda item:(-item[1][2],item[0])):
            c=cues[i]
            minimum=min(500,c.end-c.start)
            low=retimed[i-1][1] if i else 0
            high=retimed[i+1][0] if i+1<len(cues) else limit
            if (start<c.start)==reverse and low<=start<=retimed[i][1]-minimum:
                retimed[i][0]=start
            if (end>c.end)==reverse and retimed[i][0]+minimum<=end<=high:
                retimed[i][1]=end
    final=[g.Cue(a,b,c.text) for (a,b),c in zip(retimed,cues)]
    assert all(0<=x.start<x.end<=limit for x in final)
    assert all(x.end<=y.start for x,y in zip(final,final[1:]))

    counts = dict(changed=sum(a != b for a,b in zip(cues,final)), unchanged=sum(a == b for a,b in zip(cues,final)), split=0)
    counts['changes'] = [dict(refined_cue=i, original=[g.clock(old.start),g.clock(old.end)],
                             revised=[[g.clock(new.start),g.clock(new.end)]],
                             start_delta_ms=new.start-old.start, end_delta_ms=new.end-old.end)
                         for i,(old,new) in enumerate(zip(cues,final),1) if old != new]
    return final, counts
