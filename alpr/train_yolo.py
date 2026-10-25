from pathlib import Path
import os,argparse,json,shutil

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=16);p.add_argument('--size',type=int,default=640);p.add_argument('--scale',type=float,default=.3);p.add_argument('--patience',type=int,default=6);a=p.parse_args()
 out=Path(a.out).resolve();assert not out.exists(),'Choose a new output folder'
 runtime=out.parent/'yolo_runtime';(runtime/'Ultralytics').mkdir(parents=True,exist_ok=True)
 os.environ['YOLO_CONFIG_DIR']=str(runtime);os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['WANDB_DISABLED']='true'
 from ultralytics import YOLO,settings
 for key in ['sync','clearml','comet','dvc','mlflow','raytune','tensorboard','wandb']:
  if key in settings:settings.update({key:False})
 weights=Path(a.weights).resolve();assert weights.is_file()
 model=YOLO(str(weights))
 # No left/right flips: plate characters and layouts are not mirror invariant.
 result=model.train(data=str(Path(a.data).resolve()),epochs=a.epochs,imgsz=a.size,batch=a.batch,device=0,
                    project=str(out.parent),name=out.name,exist_ok=False,workers=0,cache='ram',
                    optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,patience=a.patience,
                    warmup_epochs=2,close_mosaic=5,mosaic=.5,mixup=0,fliplr=0,flipud=0,
                    degrees=5,translate=.05,scale=a.scale,perspective=0,hsv_h=.01,hsv_s=.3,hsv_v=.3,
                    seed=42,deterministic=True,amp=False,plots=False,save=True,verbose=False)
 actual=Path(model.trainer.save_dir).resolve()
 if actual!=out:
  assert not out.exists()
  shutil.copytree(actual,out)
 print('TRAINING_COMPLETE',out,flush=True)
 print(json.dumps(result.results_dict),flush=True)

if __name__=='__main__':main()
