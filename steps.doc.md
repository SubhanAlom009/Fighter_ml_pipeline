1.create a virtual environment

```bash
python -m venv venv
```

2.activate the virtual environment

```bash
# On Windows
venv\Scripts\activate
```

3.install the required packages

```bash
pip install -r requirements.txt
```

4.run the FastAPI application

```bash
uvicorn main:app --reload --port 8000
```

---
