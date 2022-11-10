import os

if __name__ == "__main__":
    # attributes = ["gravity", "masscart", "masspole", "length", "force_mag"]
    attributes = ["force_mag"]
    tt = 51200
    
    for attr in attributes:
        os.system(f'poetry run python ppo_trsnsfer.py --env-id CartPole-v1 --attribute {attr} --total-timesteps {tt} > {attr}.txt')