import asyncio
import json
import websockets

async def main():

    uri = "ws://localhost:8000/ws/client1"

    async with websockets.connect(uri) as websocket:

        print("\nCONNECTED TO SERVER\n")

        await websocket.send(

            json.dumps({

                "query":
                "Explain Kubernetes autoscaling"
            })
        )

        while True:

            response = await websocket.recv()

            data = json.loads(response)

            print(data)

            if data["type"] == "complete":

                print("\nSTREAM FINISHED\n")

                break

asyncio.run(main())