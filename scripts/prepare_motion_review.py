"""Prepare auditable contact sheets; never accept candidates automatically."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pi0_repro.quality import audit_episode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--collection',action='append',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    entries=[]
    for collection in args.collection:
        collection=collection.resolve()
        if not collection.is_relative_to(ROOT/'data/banana_demos'):raise ValueError('Outside demonstration root')
        for path in sorted(collection.glob('episode_*')):
            if not (path/'episode.json').exists():continue
            meta=json.loads((path/'episode.json').read_text());quality=None
            try:quality=audit_episode(path)
            except (ValueError,IndexError) as exc:quality={'structural_pass':False,'errors':[str(exc)]}
            n=meta['steps'];selected=[]
            if n:
                with np.load(path/'trajectory.npz',allow_pickle=False) as a:phases=a['phase']
                selected.append(('start',0))
                for title,names in (('grasp/entry',('close_gripper','descend','release')),('carry/lower',('transport','lift','lower_into_box')),('release',('release',))):
                    found=next((np.flatnonzero(phases==name) for name in names if np.any(phases==name)),np.array([0]))
                    index=int(found[-1] if title=='grasp/entry' else found[len(found)//2])
                    selected.append((title,index))
            entries.append({'id':len(entries)+1,'episode_directory':str(path),'episode':meta['episode'],
                            'layout_id':meta.get('layout_id'),'trajectory_sha256':hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest(),
                            'automatic_success':meta.get('automatic_success'),'error':meta.get('error'),
                            'quality':quality,'selected_frames':selected})
    for start in range(0,len(entries),3):
        subset=entries[start:start+3];sheet=Image.new('RGB',(5*280,len(subset)*276),'white');draw=ImageDraw.Draw(sheet)
        sheet_path=out/f'page_{start//3:02d}.jpg'
        for row,entry in enumerate(subset):
            path=Path(entry['episode_directory']);y=row*276
            header=f"ID{entry['id']:03d} {entry['layout_id']} success={entry['automatic_success']} structural={entry['quality']['structural_pass']}"
            draw.text((5,y+3),header,fill='black' if entry['quality']['structural_pass'] else 'red')
            frame_proofs=[]
            for col,(title,index) in enumerate(entry['selected_frames']):
                p=path/'rgb'/f'{index:06d}.png';im=Image.open(p).convert('RGB').crop((275,110,540,345)).resize((280,248))
                sheet.paste(im,(col*280,y+28));draw.text((col*280+4,y+16),f'{title} step={index}',fill='black')
                frame_proofs.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            if (path/'terminal_angle.png').exists():
                p=path/'terminal_angle.png';im=Image.open(p).convert('RGB');im.thumbnail((280,248));sheet.paste(im,(4*280,y+28))
                draw.text((4*280+4,y+16),'terminal angle',fill='black');frame_proofs.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            entry['frame_evidence']=frame_proofs;entry['contact_sheet']=str(sheet_path)
        sheet.save(sheet_path)
        digest=hashlib.sha256(sheet_path.read_bytes()).hexdigest()
        for entry in subset:entry['contact_sheet_sha256']=digest
    (out/'index.json').write_text(json.dumps({'scope':'Prepared evidence only; assistant must inspect actual pages before explicit acceptance. Crops match modelROI; terminal angle provides wider context.','entries':entries},indent=2)+'\n')
    print('Prepared',len(entries),'candidate rows in',(len(entries)+2)//3,'pages; none accepted.')


if __name__=='__main__':main()
