# Nutri Model for Image-Recipe Querying

Most food waste in the United States originates from households and grocery stores. One of the main reasons for this is that retailers often encourage consumers to continually buy foods leading to many unused pantry ingredients. This problem is compounded by the fact that it is difficult to synthesize balanced recipes using arbitrary leftover food
items that are already in the pantry, causing many of these materials to spoil.

We propose a food and nutrition application that gives users quick access to recipes based on the ingredients they
have at home, where users can capture and upload a photo of all their food items on the app for processing. We then
utilize object detection algorithms that can identify the ingredients in uploaded images coupled with a large recipe search engine, to recommend thousands of related recipes to the user. This is a completely new approach to recipe search and retrieval, at is entirely image-based, synthesizing a new process for recipe inspiration.

With this program, we hope to bring a reduction in food waste for users and to encourage sustainable eating with accessible ingredients in the everyday pantry.

## Installation

```bash
git https://github.com/HarvielArcilla/CS131-Final.git
cd CS131-Final
```

## Basic Usage
After installing required dependencies, the web application can be tried out by running:

```bash
python dashboard/app-dash.py
```

## File Structure
- `classification`: folder for baseline CNN model,
  - `train.ipynb`: main notebook
- `dashboard`: folder for Nutri front-end application
  - `app-dash.py`: application file
- `dataset`: folder for dataset class name and recipe-search logic
  - `recipe_search.py`: recipe search logic file
- `fridge_demo`: folder for YOLOv8 model and training
  - `demo_notebook.ipynb`: main notebook
- `object_detection`: folder for SAM testing
  - `CS131_Final.ipyn`: main notebook

This project was created for CS131 by Harviel Arcilla, Colette Do, and Brian Lee.
