# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message_from_client')
def handle_message(message):
    print('Received video feed:', message)

    # You can process the video feed here if needed
    # For example, you can save the video frames to a file or perform some analysis on the frames.

    # Send back a response to the client (optional)
    emit('message_from_server', 'Video feed received by the server.')

if __name__ == '__main__':
    socketio.run(app)
