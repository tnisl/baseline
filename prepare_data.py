"""Build train/validation data for each role using real ToolEnv replay."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lifemate_v8_finetune.training.data import main
if __name__=='__main__':main()
