# DigitalKin2

## How to install the project

1. Clone the repo
```bash
git clone git@github.com:Karine-Bauch/DigitalKin2.git
```

2. Checkout to trunk branch

3. Install dependencies
```bash
pip install -e .
```

## How to use the project in OpenApi Doc

Run the API (at the root of the project)
```bash
fastapi dev src/api/router.py
```

Got to [OpenApi Documentation](http://127.0.0.1:8000/docs)

Click on **"try it out"**
Enter a customization if you want or let it blank and click on **"Execute"**

In the response body, find the weather-appropriate recipe.