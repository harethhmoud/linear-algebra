import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output

# Initialize the Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Geometry of Linear Equations: Ax = b (2D)"),

    html.Div([
        # Sliders for Matrix A
        html.Label("Matrix A = [[a11, a12], [a21, a22]]"),
        html.Div([
            html.Label("a11:", style={'paddingRight': '10px'}),
            dcc.Slider(id='a11_slider', min=-5, max=5, step=0.5, value=2, marks={i: str(i) for i in range(-5, 6)}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
         html.Div([
            html.Label("a12:", style={'paddingRight': '10px'}),
            dcc.Slider(id='a12_slider', min=-5, max=5, step=0.5, value=1, marks={i: str(i) for i in range(-5, 6)}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
         html.Div([
            html.Label("a21:", style={'paddingRight': '10px'}),
            dcc.Slider(id='a21_slider', min=-5, max=5, step=0.5, value=1, marks={i: str(i) for i in range(-5, 6)}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
         html.Div([
            html.Label("a22:", style={'paddingRight': '10px'}),
            dcc.Slider(id='a22_slider', min=-5, max=5, step=0.5, value=3, marks={i: str(i) for i in range(-5, 6)}),
        ], style={'display': 'flex', 'alignItems': 'center'}),

        # Sliders for Vector b
        html.Label("Vector b = [b1, b2]"),
         html.Div([
            html.Label("b1:", style={'paddingRight': '22px'}),
            dcc.Slider(id='b1_slider', min=-10, max=10, step=1, value=4, marks={i: str(i) for i in range(-10, 11, 2)}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
         html.Div([
            html.Label("b2:", style={'paddingRight': '22px'}),
            dcc.Slider(id='b2_slider', min=-10, max=10, step=1, value=7, marks={i: str(i) for i in range(-10, 11, 2)}),
        ], style={'display': 'flex', 'alignItems': 'center'}),

    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

    html.Div([
        # Tabs for different views
        dcc.Tabs(id="tabs-geometry", value='tab-row', children=[
            dcc.Tab(label='Row Picture (Lines)', value='tab-row'),
            dcc.Tab(label='Column Picture (Vectors)', value='tab-column'),
        ]),
        dcc.Graph(id='linear-algebra-graph', figure={'layout': {'height': 600}}) # Set figure height
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div(id='solution-output', style={'marginTop': '20px', 'fontWeight': 'bold'})

])

# Callback to update graph and solution based on slider inputs
@app.callback(
    [Output('linear-algebra-graph', 'figure'),
     Output('solution-output', 'children')],
    [Input('a11_slider', 'value'), Input('a12_slider', 'value'),
     Input('a21_slider', 'value'), Input('a22_slider', 'value'),
     Input('b1_slider', 'value'), Input('b2_slider', 'value'),
     Input('tabs-geometry', 'value')]
)
def update_graph(a11, a12, a21, a22, b1, b2, tab):
    A = np.array([[a11, a12], [a21, a22]])
    b = np.array([b1, b2])

    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[-10, 10], constrain='domain'),
        yaxis=dict(range=[-10, 10], scaleanchor="x", scaleratio=1), # Ensure aspect ratio is 1
        title= "Row Picture" if tab == 'tab-row' else "Column Picture",
        xaxis_title="x-axis",
        yaxis_title="y-axis",
        legend_title="Components",
        height=600 # Ensure consistent height
    )
    fig.add_shape(type="line", x0=-10, y0=0, x1=10, y1=0, line=dict(color="grey", width=1))
    fig.add_shape(type="line", x0=0, y0=-10, x1=0, y1=10, line=dict(color="grey", width=1))

    solution_text = ""
    solution = None

    # Calculate determinant and attempt to find solution
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-6: # Check if matrix is singular (or nearly singular)
        # Further check if lines are coincident or parallel
        # Check if rows are proportional: a11*a22 == a12*a21
        # Check if b is also proportional: a11*b2 == a21*b1 (for first row) or a12*b2 == a22*b1 (for second row)
        # This logic needs refinement for robustness
        if abs(a11*b2 - a21*b1) < 1e-6 and abs(a12*b2 - a22*b1) < 1e-6: # Crude check for consistency (coincident)
             solution_text = f"Determinant is near zero ({det_A:.2f}). Infinite solutions (lines are likely coincident)."
        else: # Inconsistent (parallel)
             solution_text = f"Determinant is near zero ({det_A:.2f}). No unique solution (lines are likely parallel)."
    else:
        try:
            solution = np.linalg.solve(A, b)
            solution_text = f"Solution: x ≈ {solution[0]:.2f}, y ≈ {solution[1]:.2f} (Determinant: {det_A:.2f})"
        except np.linalg.LinAlgError:
             solution_text = "Error: Could not solve the system."


    # --- Plotting Logic ---
    x_vals = np.linspace(-10, 10, 100) # Range for plotting lines

    if tab == 'tab-row':
        # Plot Line 1: a11*x + a12*y = b1 => y = (b1 - a11*x) / a12 (if a12 != 0)
        if abs(a12) > 1e-6:
             y1 = (b1 - a11 * x_vals) / a12
             fig.add_trace(go.Scatter(x=x_vals, y=y1, mode='lines', name=f'{a11}x + {a12}y = {b1}'))
        else: # Vertical line: x = b1 / a11 (if a11 != 0)
            if abs(a11) > 1e-6:
                 fig.add_trace(go.Scatter(x=[b1/a11, b1/a11], y=[-10, 10], mode='lines', name=f'{a11}x = {b1} (Vertical)'))
            # else: 0x + 0y = b1 -> No line if b1 != 0, full plane if b1==0 (not easily drawn)

        # Plot Line 2: a21*x + a22*y = b2 => y = (b2 - a21*x) / a22 (if a22 != 0)
        if abs(a22) > 1e-6:
            y2 = (b2 - a21 * x_vals) / a22
            fig.add_trace(go.Scatter(x=x_vals, y=y2, mode='lines', name=f'{a21}x + {a22}y = {b2}', line=dict(dash='dash')))
        else: # Vertical line: x = b2 / a21 (if a21 != 0)
             if abs(a21) > 1e-6:
                 fig.add_trace(go.Scatter(x=[b2/a21, b2/a21], y=[-10, 10], mode='lines', name=f'{a21}x = {b2} (Vertical)', line=dict(dash='dash')))
            # else: 0x + 0y = b2

        # Plot intersection point if solution exists
        if solution is not None:
            fig.add_trace(go.Scatter(x=[solution[0]], y=[solution[1]], mode='markers', marker=dict(size=12, color='red'), name='Solution (Intersection)'))

    elif tab == 'tab-column':
        col1 = A[:, 0]
        col2 = A[:, 1]

        # Function to add vector arrows
        def add_vector(vec, name, color, origin=[0,0], dash=None):
             fig.add_trace(go.Scatter(x=[origin[0], origin[0] + vec[0]], y=[origin[1], origin[1] + vec[1]],
                                      mode='lines+markers', marker=dict(symbol='arrow', size=10, angleref='previous'),
                                      line=dict(color=color, width=2, dash=dash), name=name))

        # Plot column vectors
        add_vector(col1, f'col1 = [{a11}, {a21}]', 'blue')
        add_vector(col2, f'col2 = [{a12}, {a22}]', 'green')

        # Plot target vector b
        add_vector(b, f'b = [{b1}, {b2}]', 'red')

        # If solution exists, plot the combination
        if solution is not None:
            x_sol, y_sol = solution[0], solution[1]
            scaled_col1 = x_sol * col1
            scaled_col2 = y_sol * col2

            # Plot scaled col1
            add_vector(scaled_col1, f'{x_sol:.2f} * col1', 'lightblue', dash='dot')
            # Plot scaled col2 starting from the end of scaled_col1
            add_vector(scaled_col2, f'{y_sol:.2f} * col2', 'lightgreen', origin=scaled_col1, dash='dot')
             # Show the resultant vector (should match b) - draw slightly offset for visibility
            add_vector(scaled_col1 + scaled_col2, 'x*col1 + y*col2', 'purple', dash='longdash')


    return fig, solution_text

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) # debug=True allows live updates when code changes