import os
import time
import threading
import concurrent.futures
import signal
import sys
import Scripts.script_with_ProvenanceLLM as Script1
import Scripts.script_with_JP as Script2
from flask import Flask, render_template, jsonify
from email import message_from_bytes
from email.header import decode_header
import extract_msg

# --- Configuration ---
app = Flask(__name__)
EMAIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emails')
MAX_WORKERS = 1
PROCESSING_CACHE = {}

# NEW: Define executor in the global scope so the handler can access it
executor = None

# --- Your Custom Function & Parser ---
# (The custom_process_email_body, parse_email_file, and process_email_in_background
# functions remain unchanged from the previous version)
def custom_process_email_body(body_text: str) -> dict:
    """Processes the email body and returns a dictionary of insights."""
    print(f"Starting to process... this will take a moment.")
    time.sleep(5) # Simulate a 5-second processing time
    print("...finished processing.")
    if not body_text:
        return {"error": "No text content found to process."}
    word_count = len(body_text.split())
    char_count = len(body_text)
    long_words = [word for word in body_text.split() if len(word) > 10]
    return {
        "analysis_summary": f"Found {word_count} words and {char_count} characters.",
        "word_count": word_count,
        "character_count": char_count,
        "long_words_detected": long_words if long_words else "None"
    }

def parse_email_file(filepath):
    """Parses .eml or .msg files and returns a dictionary with email data."""
    # (code is identical to previous version)
    email_data = {}
    if filepath.endswith(".eml"):
        with open(filepath, 'rb') as f:
            msg = message_from_bytes(f.read())
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes): subject = subject.decode(encoding or "utf-8")
            email_data['subject'] = subject
            email_data['from'] = msg.get("From")
            email_data['to'] = msg.get("To")
            email_data['date'] = msg.get("Date")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                        break
                    elif part.get_content_type() == "text/plain" and not body:
                        body = f"<pre>{part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')}</pre>"
            else: body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')
            email_data['body'] = body
    elif filepath.endswith(".msg"):
        msg = extract_msg.Message(filepath)
        email_data['subject'] = msg.subject
        email_data['from'] = msg.sender
        email_data['to'] = msg.to
        email_data['date'] = msg.date
        email_data['body'] = f"<pre>{msg.body}</pre>"
    return email_data

def process_email_in_background(filename):
    """Worker function to be run in a separate thread."""
    # (code is identical to previous version)
    filepath = os.path.join(EMAIL_DIR, filename)
    try:
        email_content = parse_email_file(filepath)
        PROCESSING_CACHE[filename]['content'] = email_content
        # processed_data = custom_process_email_body(email_content.get('body', ''))
        processed_data = Script2.process_text_with_validators(email_content.get('body', ''))
        PROCESSING_CACHE[filename]['processed'] = processed_data
        PROCESSING_CACHE[filename]['status'] = 'completed'
    except Exception as e:
        PROCESSING_CACHE[filename]['status'] = 'error'
        PROCESSING_CACHE[filename]['processed'] = {'error': str(e)}

# --- NEW: Signal handler for graceful shutdown ---
def shutdown_handler(sig, frame):
    """Handles Ctrl+C and other termination signals."""
    print("\nCtrl+C received! Shutting down the thread pool...")
    if executor:
        # This tells the executor to stop accepting new tasks and finish running tasks.
        # For Python 3.9+, you can add cancel_futures=True to cancel queued tasks.
        executor.shutdown(wait=False)
    print("Exiting application.")
    sys.exit(0)

# --- Flask Routes ---
# (All Flask routes remain unchanged)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/emails')
def get_email_list():
    return jsonify([
        {'filename': f, 'status': PROCESSING_CACHE[f]['status']} 
        for f in sorted(PROCESSING_CACHE.keys())
    ])

@app.route('/api/status')
def get_all_statuses():
    return jsonify({f: {'status': PROCESSING_CACHE[f]['status']} for f in PROCESSING_CACHE})

@app.route('/api/email/<filename>')
def get_email_content(filename):
    if filename not in PROCESSING_CACHE or PROCESSING_CACHE[filename]['status'] != 'completed':
        return jsonify({"error": "Email is not ready or not found."}), 404
    
    cached_data = PROCESSING_CACHE[filename]
    return jsonify({
        "content": cached_data.get('content'),
        "processed": cached_data.get('processed')
    })


# --- Main execution block ---
if __name__ == '__main__':
    if not os.path.exists(EMAIL_DIR):
        os.makedirs(EMAIL_DIR)
    
    # Assign the executor to the global variable
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    # NEW: Register our custom shutdown handler for the SIGINT signal (Ctrl+C)
    signal.signal(signal.SIGINT, shutdown_handler)

    print(f"Starting email processing with a pool of {MAX_WORKERS} workers...")
    email_files = [f for f in os.listdir(EMAIL_DIR) if f.endswith((".eml", ".msg"))]
    
    for filename in email_files:
        if filename not in PROCESSING_CACHE:
            PROCESSING_CACHE[filename] = {'status': 'processing', 'content': None, 'processed': None}
            executor.submit(process_email_in_background, filename)
    
    print(f"Submitted {len(email_files)} jobs. Starting Flask server immediately...")
    print("Press Ctrl+C to exit.")

    app.run(debug=True)