"""Runs the AE server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import json

from ae_manager import AEManager
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()
manager = AEManager()


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds an observation into the AE model.

    Returns action taken given current observation (int)
    """

    body = await request.body()
    if not body.strip():
        manager.reset()
        return {"predictions": []}

    try:
        input_json = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

    instances = input_json.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="Request JSON must contain an 'instances' list.")

    predictions = []
    # each is a dict with one key "observation" and the value as a dictionary observation
    for instance in instances:
        observation = instance.get("observation")
        if not isinstance(observation, dict):
            raise HTTPException(
                status_code=400,
                detail="Each instance must contain an 'observation' object.",
            )
        # reset environment on a new round
        # You will have to do your own internal counting and reset your own system between rounds!
        # if observation["step"] == 0:
            # do internal resetting here
        predictions.append({"action": manager.ae(observation)})
    return {"predictions": predictions}


# ------------------------------ RESET REMOVED ------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
