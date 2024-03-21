"""

HOW TO USE:

Sample Run

python3 app-dash.py

visit http://127.0.0.1:8050/ in your web browser.
"""

from dash import Dash, dcc, html, Input, Output, State, callback_context, callback, dash_table
import base64
import time
import io
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import sys
sys.path.append("../")
from dataset.recipe_search import initialize_optimized, powerset, get_recipes_optimized, reduce
from fridge_demo.nutri_yolov8 import get_ingredients

app = Dash(__name__)
    

# ------------------------ Images ------------------------ #
logo_img = "assets/logo.png"
upload = False

# ---------------------- Constants ----------------------- #
font = 'Trebuchet MS'
DATASET_DIR = "../dataset/"
MODEL_DIR = "../fridge_demo/"
initialize_optimized(DATASET_DIR)

# ------------------------ Helper ------------------------ #
def get_df_recipes(food):
    dfs = []
    food_list = food.split(",")
    if len(food_list) >= 3:
        for i in powerset(food_list):
            if len(i) >= 3:
                dfs.append(get_recipes_optimized(i))
        return reduce(lambda x, y: x.merge(y, how='outer', on=['title','link']), dfs)
    else: 
        return get_recipes_optimized(food_list)

def link_to_embedded(link):
    return f"<a href='https://{link}' target='_blank'>{link}</a>"

# ---------------------- App Layout ---------------------- #
app.layout = html.Div(style={'display': 'flex', 
                             'flexDirection': 'column', 
                             'textAlign': 'center', 
                             'alignItems': 'center'}, 
    children=[
    html.Img(src=logo_img, style={
            'height': '50px',
        }),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'drag and drop or ',
            html.A('select files')
        ], style={'textAlign': 'center', 
                  'width': '30vw'}, hidden=upload),
        style={
            'width': '30vw',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '2vh',
            'fontFamily': font
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(children=
             [
                html.Div(id='output-image-upload'),
            ], style={'display': 'flex', 'width': '100vw', 'minHeight': '60vh', 'justifyContent': 'center', 'backgroundColor': '#38671D', 'color': 'white'})

])

# ---------------------- Callbacks ---------------------- #

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img_bytes = io.BytesIO(decoded)
    try:
        if 'png' in filename or 'jpg' in filename or 'jpeg' in filename:
            img = Image.open(img_bytes)
            print("load")
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    print("finding ingredients")
    ingredients, bounding_box = get_ingredients(MODEL_DIR, img)
    num_ingredients = len(ingredients.split(","))
    print("getting recipes")
    recipes = get_df_recipes(ingredients)
    recipes.rename(columns={'title': 'Recipe', 'link': 'Link'}, inplace=True)
    recipes['Link'] = recipes.apply(lambda x: link_to_embedded(x['Link']), axis=1)

    num_recipes = len(recipes)


    return html.Div([
        html.Div([
            html.H1(f"{num_ingredients} ingredient{'s' if num_ingredients > 1 else ''} processed", style={'fontFamily': font}),
            html.H5(filename),
            html.Img(src=bounding_box, style={
            'width': '50vw',
            }),
        ], style={'textAlign': 'center', 'width': '100%', 'height': '100%'}),
        html.Div([
            html.H1(f"{num_recipes} recipe{'s' if num_recipes > 1 else ''} found!", style={'fontFamily': font}), 
            dash_table.DataTable(
                    recipes.to_dict('records'), [{"name": i, "id": i, "presentation": "markdown"} for i in recipes.columns[0:min(len(recipes), 5)]],
                    style_cell={'overflow': 'hidden', 
        'textOverflow': 'ellipsis', 'maxWidth': 0, 'fontFamily': font, 'textAlign': 'left', 'color': 'black'},
                    style_table={'textAlign': 'center'},
                    markdown_options={"html": True}
                )
        ], style={'display': 'flex', 'flexDirection': 'column', 'width': '100%', 'textAlign': 'center', 'height': '100%'})
    ], style={'display': 'flex', 'flexDirection': 'column', 'width': '100%', 'height': '100%', 'alignItems': 'center'})


@callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# --------------------------------------------------------- #

if __name__ == '__main__':
    app.run_server(debug=True)
