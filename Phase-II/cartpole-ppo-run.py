import os

if __name__ == "__main__":
    attributes = ["gravity", "masscart", "masspole", "length", "force_mag"]
    # attributes = ["masscart"]
    tt = int(51200 * 4)
    
    factors = {
        "force_mag": 3,
        "length": 10,
        "masspole": 3,
        "masscart": 3,
        "gravity": 2,
    }
    
    for attr in attributes:
        os.system(f'poetry run python ppo_transfer.py --env-id CartPole-v1 --attribute {attr} --total-timesteps {tt} --factor {factors[attr]} --num-changes 5 > {attr}.txt')
