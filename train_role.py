"""Train exactly one role-local LoRA adapter; no global stage."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lifemate_v8_finetune.training.train import main
if __name__=='__main__':main()
