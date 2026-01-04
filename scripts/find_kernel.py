from kaggle.api.kaggle_api_extended import KaggleApi
import json

api = KaggleApi()
api.authenticate()

kernels = api.kernels_list(user="zigomaro", search="RemoteChecker", page_size=10)
for k in kernels:
    print(f"Title: {k.title}")
    print(f"Slug: {k.slug}")
    print(f"Ref: {k.ref}")
    print(f"Last Run: {k.lastRunTime}")
    print("-" * 20)
