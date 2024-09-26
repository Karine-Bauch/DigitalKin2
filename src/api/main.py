import fastapi
import services.generate_state_of_art


app = fastapi.FastAPI()

@app.get("/state_of_art")
def state_of_art(custom_details: str = "") -> str:
    try:
        response = services.generate_state_of_art.generate(custom_details)
        return response
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"An error occurred: {e}")
