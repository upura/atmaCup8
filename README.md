# #8 [初心者向] atmaCup

https://www.guruguru.science/competitions/13/

Late submission named `run008` scored public: 0.5226 and private: 0.5256.

- LightGBM
- StratifiedKFold(10) by binned target

## Environment

```sh
docker-compose -d --build
docker exec -it ayniy-test bash
```

## Run

```sh
cd experiments
python fe_basic.py
python fe_name.py
python fe_rank.py
python create_base.py
python runner.py --run configs/run008.yml
```
