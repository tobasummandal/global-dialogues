import modal

# Define the Modal app
app = modal.App("ai-personas-website")

image = (
    modal.Image.debian_slim()
    .pip_install("flask", "flask-cors", "textblob", "plotly", "gunicorn")
    .add_local_file("app.py", remote_path="/root/app.py")
    .add_local_file("persona_results.json", remote_path="/root/persona_results.json")
    .add_local_dir("templates", remote_path="/root/templates")
    .add_local_dir("static", remote_path="/root/static")
)

# Define the web app
@app.function(image=image)
@modal.web_server(8000)
def run():
    import subprocess
    
    # Start the Gunicorn server
    cmd = "gunicorn --bind 0.0.0.0:8000 app:app"
    subprocess.Popen(cmd, shell=True)
