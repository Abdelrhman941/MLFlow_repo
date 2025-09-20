<body>
    <div style = "
        width: 100%;
        height: 30px;
        background: linear-gradient(to right,rgb(235, 238, 212),rgb(235, 238, 212));">
    </div>
</body>

# **How an ML API Works**
- Client → JSON → Server → JSON back
- Sync = one at a time → Flask + Gunicorn
- Async = many at once → FastAPI + Uvicorn
- Small app → Flask
- Real app → FastAPI

| **Thing**    | **What It Does**                    | **Best For**                  |
|--------------|-------------------------------------|-------------------------------|
| **Sync**     | One request at a time.              | Small, simple apps.           |
| **Async**    | Many requests at once.              | Busy apps with lots of users. |
| **Flask**    | Easy tool for APIs (sync only).     | Beginners, small projects.    |
| **FastAPI**  | Fast, modern tool for APIs (async). | Big, real-world apps.         |
| **Gunicorn** | Runs sync APIs (e.g., Flask).       | Simple apps.                  |
| **Uvicorn**  | Runs async APIs (e.g., FastAPI).    | Fast, busy apps.              |

![API Architecture](https://i.postimg.cc/NMps5V96/image.png)

<body>
    <div style = "
        width: 100%;
        height: 30px;
        background: linear-gradient(to right,rgb(235, 238, 212),rgb(235, 238, 212));">
    </div>
</body>

# **In FastAPI**

After running the application with either:

```bash
uvicorn main:app --reload --port 8000
```

or

```bash
python main.py
```

you can access the interactive API documentation by visiting:

* **Swagger UI:** `http://127.0.0.1:8000/docs`     [add docs]
* **ReDoc:** `http://127.0.0.1:8000/redoc`         [add redoc]

---
- You can use these interfaces to **test your API endpoints interactively**.
- For example, in `main.py` go to `/docs`, click **"Try it out"**, enter a value (e.g., a name) in the input box, and then click **"Execute"**.
- The response will be displayed in the **Response body** section.

<body>
    <div style = "
        width: 100%;
        height: 30px;
        background: linear-gradient(to right,rgb(235, 238, 212),rgb(235, 238, 212));">
    </div>
</body>
