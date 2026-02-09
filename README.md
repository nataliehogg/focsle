# 🧭 focsle

FORECASTLE (FORECAST for Line-of-sight Experiments)

The best way to ensure you have the correct dependencies is to set up a virtual environment as follows:

### Install uv                                                                                                                
`curl -LsSf https://astral.sh/uv/install.sh | sh`                                                                                                        
                                                                                                                                                       
### Clone the repo  
```
git clone https://github.com/nataliehogg/focsle.git
cd focsle
uv sync
```
Or, if you don't want to use uv, you can do it this way:                                                    
                                                                                                                                                       
### Clone the repo                                                                                                                                       
```
git clone https://github.com/nataliehogg/focsle.git
cd focsle
```                                                                                                                                           
                                                                                                                                                       
### Create and activate virtual environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
                                                                                                                                                       
### Install dependencies                                                                                                                                 
`pip install -e ".[dev]"`
