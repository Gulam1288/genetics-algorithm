from flask import Flask, render_template, request, url_for, redirect,session
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import base64
import io
from math import gcd
import bcrypt
import mysql.connector

app = Flask(__name__)

data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Genetic Algorithm for Feature Selection
def initialize_population(n_population, n_features):
    return np.random.randint(2, size=(n_population, n_features))

from joblib import Parallel, delayed

# Modify fitness_function to accept only one individual at a time
def fitness_function(chromosome, X, y):
    selected_features = np.where(chromosome == 1)[0]
    if len(selected_features) == 0:
        return 0.0
    X_selected = X[:, selected_features]
    X_train, X_valid, y_train, y_valid = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    return accuracy


def select_parents(population, fitness_scores, n_parents):
    return population[np.argsort(fitness_scores)[::-1][:n_parents]]

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutate_population(population, mutation_rate):
    mutation_indices = np.random.rand(population.shape[0], population.shape[1]) < mutation_rate
    population[mutation_indices] = 1 - population[mutation_indices]
    return population

from fractions import Fraction
def scale_down(values):
    # Calculate the scaling factor using fractions to maintain ratio
    scaling_factor = Fraction(1, max(values))

    # Scale down the values while maintaining the ratio
    scaled_values = [int(value * scaling_factor) for value in values]

    # Ensure they are greater than 1
    min_value = min(scaled_values)
    if min_value <= 1:
        scale_up = 1 - min_value
        scaled_values = [max(1, value + scale_up) for value in scaled_values]

    # Ensure they are single-digit values
    max_value = max(scaled_values)
    if max_value > 9:
        scale_down = max_value // 9 + 1
        scaled_values = [value // scale_down for value in scaled_values]

    # Check if ratios are maintained
    if all(value == scaled_values[0] for value in scaled_values):
        return scaled_values
    else:
        # If ratios are not maintained, recursively scale down again
        return scale_down(scaled_values)
    
@app.route('/result', methods=['POST'])
def result():
    # Retrieve form data
    n_population = int(request.form['n_population'])
    n_generations = int(request.form['n_generations'])
    n_parents = int(request.form['n_parents'])
    mutation_rate = float(request.form['mutation_rate'])

    # Find the greatest common divisor (GCD)
    gcd_value = gcd(gcd(n_population, n_generations), n_parents)

    # Scale down the form data values by dividing them with the GCD
    n_population //= gcd_value
    n_generations //= gcd_value
    n_parents //= gcd_value

    # Find the scale factor based on the maximum value
    max_value = max(n_population, n_generations, n_parents)
    scale_factor = max_value // 9 + 1

    # Scale down each value while maintaining the ratios
    n_population //= scale_factor
    n_generations //= scale_factor
    n_parents //= scale_factor

    # Ensure the values are within the range 1-9
    n_population = max(2, max(n_population, 5))
    n_generations = max(1, max(n_generations, 2))
    n_parents = max(1, max(n_parents, 1))
    print(n_population,n_generations,n_parents)

    print(mutation_rate)

    n_features = X_train.shape[1]

    # Initialize population
    population = initialize_population(n_population, n_features)

    # Genetic Algorithm loop
    best_fitness_over_generations = []
    for _ in range(n_generations):
        # Calculate fitness scores in parallel
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(fitness_function)(chromosome, X_train, y_train) for chromosome in population
        )

        # Select parents
        parents = select_parents(population, fitness_scores, n_parents)

        # Generate offspring through crossover
        offspring = crossover(parents, offspring_size=(n_population - parents.shape[0], n_features))

        # Mutate offspring
        mutated_offspring = mutate_population(offspring, mutation_rate)

        # Create new population
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutated_offspring
        # Track best fitness (accuracy) of this generation
        best_fitness_over_generations.append(np.max(fitness_scores))

    # Select best chromosome (solution)
    best_chromosome_idx = np.argmax(fitness_scores[-1])
    best_chromosome = population[best_chromosome_idx]
    selected_features = np.where(best_chromosome == 1)[0]

    # Evaluate on test set
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    clf_final = RandomForestClassifier(random_state=42)
    clf_final.fit(X_train_selected, y_train)
    y_pred_test = clf_final.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_pred_test)


    # Plot confusion matrix with explanation
    cm = confusion_matrix(y_test, y_pred_test)
    img = io.BytesIO()
    plt.figure(figsize=(8,6))
    plt.plot(range(len(best_fitness_over_generations)), best_fitness_over_generations, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Accuracy)')
    plt.title('Evolution of Best Fitness over Generations')
    plt.grid(True)
    plt.savefig(img, format='png')
    img.seek(0)
    plot1 = base64.b64encode(img.getvalue()).decode()


    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(img, format='png')
    img.seek(0)
    plot2 = base64.b64encode(img.getvalue()).decode()


    fpr, tpr, _ = roc_curve(y_test, clf_final.predict_proba(X_test_selected)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, clf_final.predict(X_test_selected))))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(img, format='png')
    img.seek(0)
    plot3 =  base64.b64encode(img.getvalue()).decode()
    
    return render_template('result.html', test_accuracy=test_accuracy, plot1 = 'data:image/png;base64,{}'.format(plot1),plot2 = 'data:image/png;base64,{}'.format(plot2), plot3 = 'data:image/png;base64,{}'.format(plot3))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/genetic")
def genetic():
    if 'user' in session:
        user = session['user']
        return render_template("genetic.html",user=user)
    else:
        return redirect(url_for('login'))

app.secret_key = 'your_secret_key'

# Configure MySQL connection
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='gen_users'
)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        name = request.form['name']
        gender = request.form['gender']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']

        try:
            # Create cursor
            cursor = db.cursor()

            # Check if user already exists
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()

            if existing_user:
                error = "Username or email already exists. Please choose a different one."
                return render_template('register.html', error=error)

            # Insert new user with hashed password
            insert_query = "INSERT INTO users (username, password, name, gender, email, phone, address) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(insert_query, (username, hashed_password, name, gender, email, phone, address))

            # Commit to DB
            db.commit()

            return render_template('register.html', message="Registraion successful!")

        except mysql.connector.Error as err:
            error = f"An error occurred: {err}"
            return render_template('register.html', error=error)

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        # If user is already logged in, redirect to index or display a message
        message = "You are already logged in. No new logins available."
        return render_template('login.html', message=message)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            # Create cursor
            cursor = db.cursor()

            # Execute query to fetch user from database
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()

            if user:
                # Verify password
                hashed_password = user[2]  # Assuming password hash is stored in the second column
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                    # Store user information in session
                    session['user'] = user
                    return redirect(url_for('genetic'))
                else:
                    error = 'Invalid credentials. Please try again.'
                    return render_template('login.html', error=error)
            else:
                error = 'Invalid credentials. Please try again.'
                return render_template('login.html', error=error)

        except mysql.connector.Error as err:
            error = f"An error occurred: {err}"
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/reset', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['password']
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        try:
            # Create cursor
            cursor = db.cursor()

            # Check if the username exists in the database
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user:
                # Update user's password
                cursor.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password, username))
                db.commit()
                cursor.close()
                message = 'Password reset successful. You can now log in with your new password.'
                return render_template('reset.html', message=message)
            else:
                error = 'Username not found. Please enter a valid username.'
                return render_template('reset.html', error=error)

        except mysql.connector.Error as err:
            error = f"An error occurred: {err}"
            return render_template('reset.html', error=error)

    return render_template('reset.html')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0') 
