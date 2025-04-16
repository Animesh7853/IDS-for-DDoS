# Network Attack Monitoring Dashboard

A real-time dashboard for monitoring network traffic and detecting potential attacks.

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Required Dependencies

- FastAPI
- Uvicorn[standard] (with WebSocket support)
- TensorFlow
- NumPy
- Websockets

## Running the Application

You can run the application using:

```bash
python app.py
```

Or use the batch file for Windows users:

```
run_dashboard.bat
```

Then access the dashboard at: http://localhost:8000/dashboard

## Troubleshooting

If you see "Unsupported upgrade request" errors related to WebSockets, make sure you have the WebSocket libraries installed:

```bash
pip install websockets
```

or

```bash
pip install 'uvicorn[standard]'
```

## API Documentation

API documentation is available at: http://localhost:8000/docs
