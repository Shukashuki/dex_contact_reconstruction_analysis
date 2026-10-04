"""Export a locally licensed MANO model; weights are not distributed here."""
import argparse
from dataclasses import fields
from pathlib import Path
import numpy as np
from perception.mano_layer import load_mano, ManoModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--side',choices=('left','right'),default='right')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model=load_mano(a.side,a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**{f.name:getattr(model,f.name) for f in fields(ManoModel) if f.name!='side'})


if __name__=='__main__':main()
