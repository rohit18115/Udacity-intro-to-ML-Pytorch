# Here is a list of things i found worth noting:

[Cheatsheet for markdown in jupyter notebook](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

[Working with coding cells](https://github.com/rohit18115/Udacity-intro-to-ML-Pytorch/blob/main/Setup/working-with-code-cells.ipynb)
[Keyboard shortcuts](https://github.com/rohit18115/Udacity-intro-to-ML-Pytorch/blob/main/Setup/keyboard-shortcuts.ipynb)

# Magic keywords

Magic keywords are special commands you can run in cells that let you control the notebook itself or perform system calls such as changing directories. For example, you can set up matplotlib to work interactively in the notebook with `%matplotlib`.

Magic commands are preceded with one or two percent signs (`%` or `%%`) for line magics and cell magics, respectively. Line magics apply only to the line the magic command is written on, while cell magics apply to the whole cell.

**NOTE:** These magic keywords are specific to the normal Python kernel. If you are using other kernels, these most likely won't work.

## **Timing code**

At some point, you'll probably spend some effort optimizing code to run faster. Timing how quickly your code runs is essential for this optimization. You can use the `timeit` magic command to time how long it takes for a function to run, like so:

![https://video.udacity-data.com/topher/2016/November/582f354d_magic-timeit/magic-timeit.png](https://video.udacity-data.com/topher/2016/November/582f354d_magic-timeit/magic-timeit.png)

If you want to time how long it takes for a whole cell to run, you’d use `%%timeit` like so:

![https://video.udacity-data.com/topher/2016/November/58337d71_magic-timeit2/magic-timeit2.png](https://video.udacity-data.com/topher/2016/November/58337d71_magic-timeit2/magic-timeit2.png)

## **Embedding visualizations in notebooks**

As mentioned before, notebooks let you embed images along with text and code. This is most useful when you’re using `matplotlib` or other plotting packages to create visualizations. You can use `%matplotlib` to set up `matplotlib` for interactive use in the notebook. By default, figures will render in their own window. However, you can pass arguments to the command to select a specific ["backend"](http://matplotlib.org/faq/usage_faq.html#what-is-a-backend), the software that renders the image. To render figures directly in the notebook, you should use the inline backend with the command `%matplotlib inline`.

> Tip: On higher resolution screens such as Retina displays, the default images in notebooks can look blurry. Use %config InlineBackend.figure_format = 'retina' after %matplotlib inline to render higher resolution images.
> 

![https://video.udacity-data.com/topher/2016/November/5833867f_magic-matplotlib/magic-matplotlib.png](https://video.udacity-data.com/topher/2016/November/5833867f_magic-matplotlib/magic-matplotlib.png)

Example figure in a notebook

## **Debugging in the Notebook**

With the Python kernel, you can turn on the interactive debugger using the magic command `%pdb`. When you cause an error, you'll be able to inspect the variables in the current namespace.

![https://video.udacity-data.com/topher/2016/November/58337eac_magic-pdb/magic-pdb.png](https://video.udacity-data.com/topher/2016/November/58337eac_magic-pdb/magic-pdb.png)

Debugging in a notebook

Above you can see I tried to sum up a string which gives an error. The debugger raises the error and provides a prompt for inspecting your code.

Read more about `pdb` in [the documentation](https://docs.python.org/3/library/pdb.html). To quit the debugger, simply enter `q` in the prompt.

## **More reading**

There are a whole bunch of other magic commands, I just touched on a few of the ones you'll use the most often. To learn more about them, [here's the list](http://ipython.readthedocs.io/en/stable/interactive/magics.html) of all available magic commands.

# Converting notebooks

Notebooks are just big [JSON](http://www.json.org/) files with the extension `.ipynb`.

![https://video.udacity-data.com/topher/2016/November/5833887b_notebook-json/notebook-json.png](https://video.udacity-data.com/topher/2016/November/5833887b_notebook-json/notebook-json.png)

Notebook file opened in a text editor shows JSON data

Since notebooks are JSON, it is simple to convert them to other formats. Jupyter comes with a utility called `nbconvert` for converting to HTML, Markdown, slideshows, etc. The general syntax to convert a given `mynotebook.ipynb` file to another FORMAT is:

```
jupyter nbconvert --to FORMAT mynotebook.ipynb

```

The currently supported output `FORMAT` could be either of the following (ignore case):

1. HTML,
2. LaTeX,
3. PDF,
4. WebPDF,
5. Reveal.js HTML slideshow,
6. Markdown,
7. Ascii,
8. reStructuredText,
9. executable script,
10. notebook.

For example, to convert a notebook to an HTML file, in your terminal use

```
# Install the package below, if not already
pip install nbconvert
jupyter nbconvert --to html mynotebook.ipynb

```

> Note - If you wish to install any package in conda that is not available in Anaconda distribution, such as the Airbase package, use pip install airbase, instead of conda install airbase.
> 

Converting to HTML is useful for sharing your notebooks with others who aren't using notebooks. Markdown is great for including a notebook in blogs and other text editors that accept Markdown formatting.

![https://video.udacity-data.com/topher/2016/November/58338a48_nbconvert-example/nbconvert-example.png](https://video.udacity-data.com/topher/2016/November/58338a48_nbconvert-example/nbconvert-example.png)

## **Recommended Read**

As always, learn more about `nbconvert` from the [documentation](https://nbconvert.readthedocs.io/en/latest/usage.html).


