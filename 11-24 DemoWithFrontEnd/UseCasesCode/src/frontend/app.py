import os
import sys
import subprocess
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Import e configurazioni invariati

app = Flask(__name__)
app.secret_key = 'supersecretkey'
cred = credentials.Certificate('ida.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

TRAIN_SCRIPT_PATH = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/src/usecase2/2ndUseCase.py"
DATA_DIRECTORY = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/data"
TEST_SCRIPT_PATH = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/src/usecase1/1stUseCase.py"
USE_SCRIPT_PATH = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/src/usecase3/3thUseCase.py"  
PDF_RESULT_PATH_BELIEF = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/results/test_belief_pr_re_curve.pdf"
PDF_RESULT_PATH_DISBELIEF = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/results/test_disbelief_pr_re_curve.pdf"
METRICS_PATH = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/results/test_metrics.json"
UPLOAD_FOLDER = '/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/data/'

test_metrics = {}


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Per favore accedi per accedere a questa pagina.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role'] 

        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=f"{name} {surname}",
            )

            auth.set_custom_user_claims(user.uid, {'role': role})

            db.collection('users').document(user.uid).set({
                'name': name,
                'surname': surname,
                'email': email,
                'role': role
            })

            flash('Registrazione completata! Per favore accedi.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Exception during registration: {e}")
            flash(f"Errore durante la registrazione: {e}", "error")
            return render_template('register.html')
    else:
        return render_template('register.html')
    


@app.route('/uploaddataset', methods=['GET', 'POST'])
def uploaddataset():
    if request.method == 'POST':
        # Ensure the directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Define file fields and target filenames
        files = {
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'use_file': 'test.csv'
        }

        uploaded_files = []
        for field, filename in files.items():
            file = request.files.get(field)
            if file:
                # Save each file with the new name in the specified directory
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                uploaded_files.append(filename)  # Track uploaded files

        if uploaded_files:
            flash(f"File caricati con successo: {', '.join(uploaded_files)}", "success")
        else:
            return "No files selected for upload."

    return render_template("uploaddataset.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            import requests
            FIREBASE_API_KEY = 'AIzaSyDTKhs3plCYlp1Bx2rKWG8AYrxqk7hRHco'  
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            r = requests.post(url, json=payload)
            if r.status_code == 200:
                id_token = r.json()['idToken']
                user = auth.verify_id_token(id_token)
                uid = user['uid']

                user_doc = db.collection('users').document(uid).get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    role = user_data.get('role', 'Tester') 
                else:
                    role = 'Tester'

                session['user'] = {
                    'uid': uid,
                    'email': email,
                    'role': role
                }
                flash('Accesso effettuato con successo!', 'success')
                return redirect(url_for('home'))
            else:
                error_message = r.json()['error']['message']
                flash(f"Accesso fallito: {error_message}", "error")
                return render_template('login.html')
        except Exception as e:
            print(f"Exception during login: {e}")
            flash(f"Errore durante l'accesso: {e}", "error")
            return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Disconnessione effettuata con successo!', 'success')
    return redirect(url_for('login'))
@app.route("/run_script", methods=["POST"])
@login_required
def run_script():
    global test_metrics

    action = request.form.get("action")
    model_type = request.form.get("model_type")
    user_role = session['user']['role']

    # Role-based access control
    if action == "train" and user_role not in ["Programmatore", "Admin"]:
        flash("Non hai il permesso per eseguire il training.", "error")
        return redirect(url_for("home"))
    elif action == "test" and user_role not in ["Programmatore", "Admin"]:
        flash("Non hai il permesso per eseguire il test.", "error")
        return redirect(url_for("home"))
    elif action == "use" and user_role not in ["Tester", "Admin"]:
        flash("Non hai il permesso per eseguire l'utilizzo del modello.", "error")
        return redirect(url_for("home"))

    if action not in ["train", "test", "use"] or model_type not in ["logistic", "mlp"]:
        flash("Selezione non valida!", "error")
        return redirect(url_for("home"))

    os.environ["MODEL_TYPE"] = model_type

    # Seleziona lo script appropriato in base all'azione
    if action == "train":
        script_path = TRAIN_SCRIPT_PATH
    elif action == "test":
        script_path = TEST_SCRIPT_PATH
    elif action == "use":
        script_path = USE_SCRIPT_PATH

    try:
        subprocess.check_call([sys.executable, script_path])
        flash(f"{action.capitalize()} eseguito correttamente con modello {model_type}!", "success")

        # Load appropriate metrics file based on action
        if action == "test" and os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                test_metrics = json.load(f)
            flash("Il test è stato completato. Puoi visualizzare i risultati PDF e le misure qui sotto.", "success")
        elif action == "use":
            thresholds_path = "/Users/gennarojuniorpezzullo/IDA/ida/UseCasesCode/results/thresholds_and_consistency.json"
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    test_metrics = json.load(f)
                flash("Il modello è stato utilizzato correttamente. Le soglie sono disponibili qui sotto.", "success")
            else:
                flash("File thresholds_and_consistency.json non trovato.", "error")
    except subprocess.CalledProcessError:
        flash(f"Errore durante l'esecuzione dello script {action}!", "error")

    return redirect(url_for("home"))
@app.route("/")
@login_required
def home():
    pdf_exists_belief = os.path.exists(PDF_RESULT_PATH_BELIEF)
    pdf_exists_disbelief = os.path.exists(PDF_RESULT_PATH_DISBELIEF)
    return render_template(
        "index.html",
        pdf_exists_belief=pdf_exists_belief,
        pdf_exists_disbelief=pdf_exists_disbelief,
        test_metrics=test_metrics
    )



@app.route("/view_pdf/<pdf_type>")
@login_required
def view_pdf(pdf_type):
    if pdf_type == "belief" and os.path.exists(PDF_RESULT_PATH_BELIEF):
        return send_file(PDF_RESULT_PATH_BELIEF, as_attachment=False)
    elif pdf_type == "disbelief" and os.path.exists(PDF_RESULT_PATH_DISBELIEF):
        return send_file(PDF_RESULT_PATH_DISBELIEF, as_attachment=False)
    else:
        flash("Il file PDF richiesto non è disponibile.", "error")
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)