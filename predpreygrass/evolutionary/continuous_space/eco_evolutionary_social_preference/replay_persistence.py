"""Render saved position traces; this does not run inference or change a world."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np

from .movement_metrics import wrapped_offset


def edges(frame, size, radius=100):
    positions = list(frame['predators'].values())
    lines = []
    for i, start in enumerate(positions):
        for end in positions[i+1:]:
            delta = wrapped_offset(start, end, size)
            if np.linalg.norm(delta) <= radius:
                for dx in (-size[0], 0, size[0]):
                    for dy in (-size[1], 0, size[1]):
                        a = np.asarray(start) + (dx,dy)
                        b = a + delta
                        if (np.maximum(a,b) >= 0).all() and (np.minimum(a,b) <= size).all():
                            lines.append([a,b])
    return lines


def points(frame, kind):
    return np.array(list(frame[kind].values())).reshape(-1,2)


def trails(frames, index, size, length=8):
    segments=[]
    for before, after in zip(frames[max(0,index-length):index],frames[max(0,index-length)+1:index+1]):
        for agent in before['predators'].keys() & after['predators'].keys():
            a=np.asarray(before['predators'][agent]); b=np.asarray(after['predators'][agent])
            if (np.abs(b-a) < np.asarray(size)/2).all():
                segments.append([a,b])
    return segments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--seed',type=int,default=62)
    args=parser.parse_args()
    traces=[]
    metadata=json.loads(next(args.directory.glob('neutral_s*.json')).read_text())
    # Read arena dimensions from checkpoint settings pinned by the experiment.
    settings_paths=[Path(p) for p in metadata['input_sha256'] if p.endswith('source_code/settings.json') and metadata['input_sha256'][p]]
    if not settings_paths:
        raise ValueError('this replay requires the saved run settings')
    settings_bytes=settings_paths[0].read_bytes()
    if hashlib.sha256(settings_bytes).hexdigest()!=metadata['input_sha256'][str(settings_paths[0])]:
        raise ValueError('checkpoint settings changed since recording')
    config=json.loads(settings_bytes)['config_env']
    size=np.array([config['width'],config['height']])
    for condition in ('neutral','attraction2'):
        with gzip.open(args.directory/'traces'/f'{condition}_seed{args.seed}.jsonl.gz','rt') as stream:
            frames=[json.loads(line) for line in stream]
        if any(b['tick']<=a['tick'] for a,b in zip(frames,frames[1:])):
            raise ValueError('trace ticks are not ascending')
        traces.append(frames)
    if traces[0][0] != traces[1][0]:
        raise ValueError('matched conditions have different initial positions')
    movie=args.directory/f'matched_seed{args.seed}.mp4'
    snapshots=args.directory/f'matched_seed{args.seed}_snapshots.png'
    if movie.exists() or snapshots.exists():
        raise FileExistsError('replay output already exists')
    fig,axes=plt.subplots(1,2,figsize=(10,5.6))
    artists=[]
    for ax,title in zip(axes,('Neutral','Attraction, strength 2')):
        ax.set(xlim=(0,size[0]),ylim=(size[1],0),aspect='equal',title=title,xlabel='world x',ylabel='world y')
        links=LineCollection([],colors='#159a65',linewidths=1,alpha=.6)
        tail=LineCollection([],colors='#ed8a23',linewidths=.8,alpha=.3)
        ax.add_collection(links); ax.add_collection(tail)
        prey=ax.scatter([],[],s=8,color='#4185b4',alpha=.65)
        predators=ax.scatter([],[],s=35,color='#dc6525',edgecolors='white',linewidths=.4)
        label=ax.text(.02,.98,'',transform=ax.transAxes,va='top',fontsize=9)
        artists.append((links,tail,prey,predators,label))
    fig.suptitle(f'Matched seed {args.seed} | actual positions, toroidal world')
    fig.text(.5,.025,'Orange: predators; blue: prey. Green: same-frame pairs within 100 world units. Trails span up to 64 ticks.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94])
    ticks=sorted({frame['tick'] for trace in traces for frame in trace})
    indices=[0,0]
    writer=FFMpegWriter(fps=20,codec='libx264',bitrate=1600,extra_args=['-threads','1','-pix_fmt','yuv420p'])
    with writer.saving(fig,str(movie),dpi=100):
        for tick in ticks:
            for k,(frames,art) in enumerate(zip(traces,artists)):
                while indices[k]+1<len(frames) and frames[indices[k]+1]['tick']<=tick:
                    indices[k]+=1
                frame=frames[indices[k]]
                links,tail,prey,predators,label=art
                links.set_segments(edges(frame,size))
                tail.set_segments(trails(frames,indices[k],size))
                prey.set_offsets(points(frame,'prey'));predators.set_offsets(points(frame,'predators'))
                ended=tick>frames[-1]['tick']
                label.set_text(f"tick {frame['tick']} | predators {len(frame['predators'])}"+ (' | episode ended' if ended else ''))
            writer.grab_frame()
    plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(10,18))
    for row,tick in enumerate((0,2000,4000,6000)):
        for col,(frames,title) in enumerate(zip(traces,('Neutral','Attraction 2'))):
            frame=min(frames,key=lambda f:abs(f['tick']-tick))
            ax=axes[row,col]
            ax.set(xlim=(0,size[0]),ylim=(size[1],0),aspect='equal',title=f"{title} | tick {frame['tick']} | N={len(frame['predators'])}")
            ax.add_collection(LineCollection(edges(frame,size),colors='#159a65',linewidths=1,alpha=.65))
            ax.scatter(*points(frame,'prey').T,s=8,color='#4185b4',alpha=.65)
            ax.scatter(*points(frame,'predators').T,s=30,color='#dc6525')
            for agent,(x,y) in frame['predators'].items():
                ax.text(x+5,y,str(agent).removeprefix('predator_'),fontsize=6)
    fig.suptitle(f'Preselected seed {args.seed}: orange predators, blue prey, green proximity links (100 units)')
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(snapshots,dpi=120);plt.close(fig)
    print(movie);print(snapshots)


if __name__=='__main__':
    main()
