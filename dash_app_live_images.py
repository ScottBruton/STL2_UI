import base64
import io
import os
import time
import queue
import threading
from collections import deque

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
# === CONFIGURATION ===
MAX_BATCH = 5
batch_predictions = deque(maxlen=MAX_BATCH)
batch_images = deque(maxlen=MAX_BATCH)

# === Dummy Serial Class (folder looping version) ===
class DummySerial:
    def __init__(self, folder_path, interval=2):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.interval = interval
        self.buffer = queue.Queue()
        self.running = True
        self.index = 0
        threading.Thread(target=self.feed_data, daemon=True).start()

    def feed_data(self):
        while self.running:
            try:
                img_path = self.image_paths[self.index]
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                size = len(img_bytes)
                self.buffer.put(f"START:{size}\n".encode())  # header
                for i in range(0, size, 1024):               # chunk image bytes
                    self.buffer.put(img_bytes[i:i+1024])

                time.sleep(self.interval)
                self.index = (self.index + 1) % len(self.image_paths)
            except Exception as e:
                print("DummySerial error:", e)
                self.running = False

    def readline(self):
        try:
            line = b""
            while not line.endswith(b"\n"):
                chunk = self.buffer.get(timeout=1)
                line += chunk
            return line
        except queue.Empty:
            return b""

    def read(self, size):
        data = bytearray()
        while len(data) < size:
            try:
                chunk = self.buffer.get(timeout=1)
                data += chunk
            except queue.Empty:
                break
        return data

    def close(self):
        self.running = False

# === Load TensorFlow Lite Models ===
interpreter1 = tf.lite.Interpreter(model_path="production_retrained.tflite")
# interpreter2 = tf.lite.Interpreter(model_path="production_quant.tflite")
interpreter1.allocate_tensors()
# interpreter2.allocate_tensors()

input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()
input_index1 = input_details1[0]['index']
output_index1 = output_details1[0]['index']
input_shape = input_details1[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# input_details2 = interpreter2.get_input_details()
# output_details2 = interpreter2.get_output_details()
# input_index2 = input_details2[0]['index']
# output_index2 = output_details2[0]['index']

legend = {0: "Connected", 1: "Disconnected", 2: "No SUDS", 3: "ERROR"}

ser = DummySerial("downsampled_images", interval=1)

# === Global Variables ===
running = False
serial_thread_handle = None
latest_img_src = None
latest_img_src_bounding = None
latest_label = ""
PRIMED = False
PAUSED = False  # New state variable for pause status

# External stylesheets
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Bayer Radiology: Stellinity 2.0 SUDS Detector</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #F9F9F9;
                color: #1A1A1A;
                padding: 2rem;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-family: 'Roboto', sans-serif;
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 2rem;
                background-color: white;
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .feed-container {
                display: flex;
                justify-content: space-between;
                gap: 2rem;
                margin-bottom: 2rem;
            }
            .feed-box {
                flex: 1;
                background-color: white;
                border-radius: 12px;
                border: 1px solid #E0E0E0;
                box-shadow: 0 0 12px rgba(0,0,0,0.05);
                padding: 0.5rem;
                align-items: center;
            }
            .feed-box h3 {
                font-size: 1rem;
                background-color: #ECECEC;
                padding: 0.4rem;
                margin: 0;
                border-radius: 8px 8px 0 0;
                text-align: center;
            }
            .image {
                width: 100%;
                border-radius: 0 0 8px 8px;
            }
            .button-container {
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin: 2rem 0;
            }
            .prime-btn {
                background-color: #F58220 !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
            }
            .stop-btn {
                background-color: #D32F2F !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
            }
            .status-container {
                text-align: center;
                margin-top: 2rem;
                padding: 1rem;
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .status-label {
                font-size: 1.2rem;
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            .status-pill {
                display: inline-block;
                padding: 0.5rem 2.5rem;
                border-radius: 999px;
                font-weight: 500;
                font-size: 1.2rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.10);
                transition: background 0.2s, color 0.2s;
                text-align: center;
            }
            .connected {
                background: #DFF6E3;   /* light green */
                color: #00C853;
            }
            .disconnected {
                background: #FFD6D6;   /* light red */
                color: #D32F2F;
            }
            .no-suds {
                background: #F0F0F0;   /* light gray */
                color: #616161;
            }
            .error {
                background: #FFF3E0;   /* light orange */
                color: #FFAB00;
            }
            .detecting {
                background: #FFF3E0;   /* light orange */
                color: #FFAB00;
            }
            .substatus {
                font-size: 0.9rem;
                color: #666;
                margin-top: 0.5rem;
            }
            .pause-btn {
                background-color: #2196F3 !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
            }
            .resume-btn {
                background-color: #4CAF50 !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
            }
        </style>
        <link rel="icon" type="image/png" href="/assets/bayer-logo.png"/>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Bayer Radiology: Stellinity 2.0 SUDS Detector", 
                style={"margin": "0", "fontSize": "2rem", "fontWeight": "bold", "textAlign": "center", "width": "100%"}),
        html.Img(src="/assets/bayer-logo.png", style={"height": "48px"})
    ], className="header"),

    # Hidden store for model output visibility
    dcc.Store(id='model-output-visibility', data='stopped'),
    # Store for detecting state
    dcc.Store(id='detecting-state', data='ready'),
    # Store for error modal state
    dcc.Store(id='error-modal-store', data={"show": False, "state": ""}),

    # Main Content Area
    html.Div([
        # Left Column - Camera Feed
        html.Div([
            html.H3("Camera Feed", style={"margin": "0", "padding": "0.5rem", "textAlign": "center", "width": "100%"}),
            html.Img(id="live-image-raw", style={"width": "100%", "borderRadius": "8px"})
        ], className="feed-box"),

        # Right Column - Model Detection
        html.Div([
            html.H3("Model Detection Output", id="model-detection-header", style={"margin": "0 auto", "padding": "0.5rem", "textAlign": "center", "width": "100%", "display": "block"}),
            html.Div(id="model-detection-output-box")
        ], id="model-detection-container", className="feed-box")
    ], className="feed-container"),

    # Control Buttons
    html.Div([
        dbc.Button("PRIME", id="prime-btn", className="prime-btn", n_clicks=0),
        dbc.Button("STOP", id="stop-btn", className="stop-btn", n_clicks=0, disabled=True)
    ], className="button-container"),

    # Status Output
    html.Div([
        html.Div("Detected State:", className="status-label"),
        html.Div(id="label-output", className="status-pill"),
        html.Div("SUDS 2.0 System Checkpoint: Priming AI Detection", className="substatus")
    ], className="status-container"),

    # Error Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("System Error State Detected"), close_button=False, style={"backgroundColor": "#FFF3E0", "color": "#FFAB00", "fontWeight": "bold"}),
        dbc.ModalBody([
            html.Div(id="error-modal-message", style={"fontSize": "1.2rem", "textAlign": "center", "marginBottom": "1rem"}),
            html.Div([
                dbc.Button("Retry", id="error-modal-retry", color="primary", className="me-2", style={"backgroundColor": "#F58220", "border": "none"}),
                dbc.Button("Cancel", id="error-modal-cancel", color="secondary", style={"backgroundColor": "#D32F2F", "border": "none"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "1rem"})
        ]),
    ], id="error-modal", is_open=False, centered=True, backdrop="static", style={"borderRadius": "16px", "boxShadow": "0 4px 24px rgba(0,0,0,0.15)"}),

    # Hidden intervals
    dcc.Interval(id="GUI-interval", interval=2000, n_intervals=0),
    dcc.Interval(id="image-interval-raw", interval=2000, n_intervals=0),
    dcc.Interval(id="image-interval-bounding", interval=2000, n_intervals=0),
    dcc.Interval(id="prediction-interval", interval=2000, n_intervals=0, disabled=True)
], style={"maxWidth": "1200px", "margin": "0 auto"})

def process_image(data):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_resized = img.resize((input_width, input_height))
        img_np = np.array(img_resized)

        # Prepare input for TFLite model 1
        if input_details1[0]['dtype'] == np.uint8:
            img_input1 = (img_np / 255.0 * 255).astype(np.uint8)
        else:
            img_input1 = img_np.astype(np.float32)
        img_input1 = np.expand_dims(img_input1, axis=0)

        interpreter1.set_tensor(input_index1, img_input1)
        interpreter1.invoke()
        output1 = interpreter1.get_tensor(output_index1)
        prediction1 = np.argmax(output1[0])
        label1 = legend.get(prediction1, "Unknown")
       
        return img, label1
    except Exception as e:
        return None, f"Error: {e}"
    
def enhance_contrast_clahe(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

 
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gamma_corrected)
    

    clahe_normalized = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(clahe_normalized, (3, 3), 0)
    blended = cv2.addWeighted(gray, 0.3, blurred, 0.7, 0)
    # ret, thresh1 = cv2.threshold(blended, 85,120, cv2.THRESH_BINARY)
    # ret,thresh2 = cv2.threshold(blended, 0, 50, cv2.THRESH_BINARY_INV)
    thresh1 = 40
    thresh2 = 100
    mask = cv2.inRange(blended, thresh1, thresh2)
    mask[:60, :] = 0
    mask[-15:, :] = 0     # bottom
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    img_h, img_w = img_rgb.shape[:2]
    padding = 10

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            x_pad = max(x - padding, 0)
            y_pad = max(y - padding, 0)
            w_pad = min(x + w + padding, img_w) - x_pad
            h_pad = min(y + h + padding, img_h) - y_pad
            bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))

    return bounding_boxes

# # === Serial Thread ===

def serial_thread():
    global running, latest_img_src, latest_label, latest_img_src_bounding
    running = True
    while running:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line.startswith("START:"):
                continue
            size = int(line.split(":")[1])
            data = bytearray()
            while len(data) < size:
                chunk = ser.read(size - len(data))
                if not chunk:
                    break
                data += chunk

            img, label1 = process_image(data)

            if img:
                # --- Raw image ---
                img_raw_resized = img.resize((320, 240))
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                latest_img_src = f"data:image/jpeg;base64,{img_base64}"

                # --- Processed image with bounding box ---
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  
                bounding_boxes = enhance_contrast_clahe(img_cv)
                img_with_boxes = img_cv.copy()
                for (x, y, w, h) in bounding_boxes:
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                processed_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

                processed_pil = Image.fromarray(processed_rgb).resize((320, 240))
                buffered_bounding = io.BytesIO()
                processed_pil.save(buffered_bounding, format="JPEG")
                img_bounding_base64 = base64.b64encode(buffered_bounding.getvalue()).decode()
                latest_img_src_bounding = f"data:image/jpeg;base64,{img_bounding_base64}"

                # Always update latest_label
                batch_predictions.append(label1)
                batch_images.append(img)
                for i in range(len(batch_predictions)):
                    if (batch_predictions[i] != batch_predictions[i-1]) and (len(batch_predictions) > 1):
                        print("model detecting new state...")
                        latest_label = "Detecting new state..."
                    else:
                        latest_label = label1
        except Exception as e:
            print("Thread error:", e)
            continue
    # return "" , {"color": "white", "textAlign": "center"}

serial_thread_handle = threading.Thread(target=serial_thread, daemon=True)
serial_thread_handle.start()

@app.callback(
    Output("prime-btn", "disabled"),
    Output("prime-btn", "children"),
    Output("stop-btn", "disabled"),
    Output('model-output-visibility', 'data'),
    Output('detecting-state', 'data'),
    Input('prime-btn', 'n_clicks'),
    Input('stop-btn', 'n_clicks'),
    State('model-output-visibility', 'data'),
    State('detecting-state', 'data'),
    prevent_initial_call=True
)
def control_buttons(prime_clicks, stop_clicks, current_state, detecting_state):
    ctx = callback_context
    if not ctx.triggered:
        return False, "PRIME", True, current_state, detecting_state
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'prime-btn':
        if detecting_state == 'ready':
            return True, "Priming...", True, current_state, 'detecting'
        elif detecting_state == 'detecting':
            return True, "Priming...", False, 'primed', 'ready'
    elif button_id == 'stop-btn':
        return False, "PRIME", True, 'stopped', 'ready'
    
    return False, "PRIME", True, current_state, detecting_state

@app.callback(
    Output("model-detection-container", "style"),
    Output("model-detection-header", "style"),
    Input('model-output-visibility', 'data')
)
def update_model_detection_container_style(visibility):
    if visibility == 'primed':
        return {
            "backgroundColor": "orange",
            "color": "white",
            "borderRadius": "12px",
            "border": "1px solid #E0E0E0",
            "boxShadow": "0 0 12px rgba(0,0,0,0.05)",
            "padding": "0.5rem"
        }, {"margin": "0", "padding": "0.5rem", "color": "white", "backgroundColor": "orange"}
    else:
        return {
            "backgroundColor": "white",
            "color": "#1A1A1A",
            "borderRadius": "12px",
            "border": "1px solid #E0E0E0",
            "boxShadow": "0 0 12px rgba(0,0,0,0.05)",
            "padding": "0.5rem"
        }, {"margin": "0", "padding": "0.5rem", "color": "#1A1A1A", "backgroundColor": "white"}

@app.callback(
    Output("model-detection-output-box", "children"),
    Output("label-output", "children"),
    Output("label-output", "className"),
    Output('error-modal-store', 'data'),
    Input("image-interval-bounding", "n_intervals"),
    Input('model-output-visibility', 'data'),
    Input('detecting-state', 'data'),
    State('error-modal-store', 'data')
)
def update_model_detection_output(n, visibility, detecting_state, error_modal_data):
    if error_modal_data and error_modal_data.get("show", False):
        state = error_modal_data.get("state", "")
        return (
            html.Img(src=latest_img_src_bounding, style={"width": "100%", "borderRadius": "8px"}),
            state,
            "status-pill error",
            error_modal_data
        )
    if detecting_state == 'detecting':
        return (
            html.Div(
                "Priming flag not detected.",
                style={
                    "width": "100%",
                    "height": "300px",
                    "background": "#e0e0e0",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "color": "#444",
                    "fontWeight": "bold",
                    "fontSize": "1.2rem",
                    "borderRadius": "8px"
                }
            ),
            "Detecting State...",
            "status-pill detecting",
            {"show": False, "state": ""}
        )
    elif visibility == 'primed':
        # If the detected state is not Connected or detecting, show error modal
        if latest_label not in ["Connected", "Detecting State..."]:
            return (
                html.Img(src=latest_img_src_bounding, style={"width": "100%", "borderRadius": "8px"}),
                latest_label,
                "status-pill error",
                {"show": True, "state": latest_label}
            )
        status_class = "status-pill "
        if latest_label == "Connected":
            status_class += "connected"
        elif latest_label == "Disconnected":
            status_class += "disconnected"
        elif latest_label == "No SUDS":
            status_class += "no-suds"
        else:
            status_class += "error"
        return (
            html.Img(src=latest_img_src_bounding, style={"width": "100%", "borderRadius": "8px"}),
            latest_label,
            status_class,
            {"show": False, "state": ""}
        )
    else:
        return (
            html.Div(
                "Priming flag not detected.",
                style={
                    "width": "100%",
                    "height": "300px",
                    "background": "#e0e0e0",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "color": "#444",
                    "fontWeight": "bold",
                    "fontSize": "1.2rem",
                    "borderRadius": "8px"
                }
            ),
            "System Ready",
            "status-pill",
            {"show": False, "state": ""}
        )

# Modal visibility and message
@app.callback(
    Output("error-modal", "is_open"),
    Output("error-modal-message", "children"),
    Input('error-modal-store', 'data')
)
def show_error_modal(error_modal_data):
    if error_modal_data and error_modal_data.get("show", False):
        return True, f"Detected State: {error_modal_data.get('state', '')}"
    return False, ""

# Cancel button resets everything to initial state
@app.callback(
    Output('model-output-visibility', 'data', allow_duplicate=True),
    Output('detecting-state', 'data', allow_duplicate=True),
    Output('error-modal-store', 'data', allow_duplicate=True),
    Output('prime-btn', 'disabled', allow_duplicate=True),
    Output('prime-btn', 'children', allow_duplicate=True),
    Output('stop-btn', 'disabled', allow_duplicate=True),
    Input('error-modal-cancel', 'n_clicks'),
    prevent_initial_call=True
)
def cancel_error_modal(n_clicks):
    if n_clicks:
        return 'stopped', 'ready', {"show": False, "state": ""}, False, "PRIME", True
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("live-image-raw", "src"),
    Input("image-interval-raw", "n_intervals")
)
def update_camera_image(n):
    return latest_img_src

@app.callback(
    Output('detecting-state', 'data', allow_duplicate=True),
    Output('model-output-visibility', 'data', allow_duplicate=True),
    Output('stop-btn', 'disabled', allow_duplicate=True),
    Input('detecting-state', 'data'),
    prevent_initial_call=True
)
def handle_detecting_delay(state):
    if state == 'detecting':
        time.sleep(0.3)  # 0.3 second delay
        return 'ready', 'primed', False  # Enable the stop button
    return state, dash.no_update, dash.no_update

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)

    
