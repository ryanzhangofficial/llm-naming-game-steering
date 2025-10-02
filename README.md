# SIGN: Schema Induced Games for Naming
A naming game that examines how lightweight structure can steer convention formation. We compare schema induced connmucation to unconstrained natural language and find faster convergence with up to 5.8x higher agreement.
## Minimal Setup & Run

```bash
# clone repo
git clone https://github.com/ryanzhangofficial/llm-naming-game-steering.git
cd llm-naming-game-steering

# install deps
pip install -r requirements.txt

# quick test (mock, no model)
python runner.py --condition nl --mock

# run with local model
python runner.py --model-path /path/to/model.gguf --condition schema --population-size 24 --rounds 100

# ablation run
python runner.py --model-path /path/to/model.gguf --ablation --ablation-populations 12,24 --ablation-memory 5,10 --ablation-alpha 0.5,0.75,0.9
