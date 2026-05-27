from pathlib import Path
from IPython.display import display
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

try:
    import anywidget
    import ipywidgets as widgets
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    print("Plotly FigureWidget is available.")
except Exception as error:
    widgets = None
    go = None
    PLOTLY_AVAILABLE = False
    print("Interactive Plotly viewer is unavailable in this environment.")
    print("Install these packages locally and restart the kernel:")
    print("pip install plotly anywidget ipywidgets")
    print(f"Import error: {error}")


def show_curve_for_pixel(
    row_index: int,
    col_index: int,
    hertz,
) -> None:
    """Static fallback: plot the force curve for one pixel.
    
    Parameters
    ----------
    row_index : int
        Row index of the pixel to display.
    col_index : int
        Column index of the pixel to display.
        
    Returns
    -------
    None
        Displays a plot of the force curve for the specified pixel.
        
    """

    displacement_curves_um = test_collection["displacement_curves_tip_um"]
    displacement_curve_um = \
      displacement_curves_um[row_index, col_index]
    curves_nN = test_collection["measured_force_curves_nN"]
    curve_nN = curves_nN[row_index, col_index]
    curve_lengths = test_collection["curve_lengths"]
    valid_length = int(curve_lengths[row_index, col_index])
    retract_indices = test_collection["retract_start_indices"]
    retract_start = int(retract_indices[row_index, col_index])
    stiffness_maps_kpa = test_collection["stiffness_maps_kpa"]
    true_kpa = stiffness_maps_kpa[row_index, col_index]
    pred_kpa = predicted_stiffness_maps_kpa[row_index, col_index]
    title = (
        f"Pixel ({row_index}, {col_index}) | "
        f"true = {true_kpa:.2f} kPa | pred = {pred_kpa:.2f} kPa"
    )
    if model_config.predict_contact_point and \
      predicted_contact_point_maps_nm is not None:
        contact_point_maps = test_collection["contact_point_maps_nm"]
        cp_true = contact_point_maps[row_index, col_index]
        cp_pred = \
        predicted_contact_point_maps_nm[row_index, col_index]
        title += f" | cp true = {cp_true:.1f} nm | cp pred = {cp_pred*1e9:.1f} nm"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(displacement_curve_um[:retract_start], curve_nN[:retract_start], \
            label="Approach", linewidth=2.2, alpha=0.5)
    ax.plot(
        displacement_curve_um[retract_start:valid_length],
        curve_nN[retract_start:valid_length],
        label="Retract",
        linewidth=2.2,
    )
    ax.plot(displacement_curve_um[:valid_length], 
            hertz * 1e9, 
            label="Hertz fit from pred (cp, E)", 
            linewidth=3
    )
    ax.set_xlabel("Displacement [um]", fontsize=12)
    ax.set_ylabel("Measured force [nN]", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()

def display_clickable_stiffness_map(
    map_width: int = 560,
    map_height: int = 560,
    curve_width: int = 980,
    curve_height: int = 560,
    test_collection=None,
    example_stiffness_map_pa = None,
    predicted_stiffness_map_pa = None,
    approach_force = None,
    predicted_contact_point_map_m = None,
    contact_model = None,
    simulation_config = None,
    abstract_figure: bool = False,
    save_dir: str = "results",
) -> None:
    """Display a clickable predicted stiffness map linked to a force-curve
    panel.
    
    Parameters
    ----------
    map_width : int, optional
        Width of the stiffness map panel in pixels (default is 560).
    map_height : int, optional
        Height of the stiffness map panel in pixels (default is 560).
    curve_width : int, optional
        Width of the force curve panel in pixels (default is 980).
    curve_height : int, optional
        Height of the force curve panel in pixels (default is 560).
    test_collection : dict, optional
        Dictionary containing test data, including measured force curves,
        displacement curves, curve lengths, retract start indices, and stiffness maps (default is None).
    example_stiffness_map_pa : np.ndarray, optional
        2D array of true stiffness values in Pascals for the example sample (default is None).
    predicted_stiffness_map_pa : np.ndarray, optional
        2D array of predicted stiffness values in Pascals for the example sample (default is None).
    approach_force : np.ndarray, optional
        2D array of predicted approach forces in Newtons for the example sample (default is None).
    predicted_contact_point_map_m : np.ndarray, optional
        2D array of predicted contact point values in meters for the example sample (default is None).
    contact_model : ContactModel, optional
        Contact model to use for force calculation (default is None).
    simulation_config : SimulationConfig, optional
        Simulation configuration containing tip shape and other parameters (default is None).
    abstract_figure : bool, optional
        When True, each clicked pixel also saves a minimal matplotlib figure
        (approach curve + predicted Hertz fit + GT Hertz fit) with a
        transparent background and no axes to ``save_dir`` (default is False).
    save_dir : str, optional
        Directory in which abstract figures are saved (default is "results").
            
    Returns
    -------
    None
        Displays a clickable stiffness map linked to a force-curve panel.

    """

    if not PLOTLY_AVAILABLE:
        print("Interactive viewer not available. Using static fallback.")
        show_curve_for_pixel(0, 0, contact_model, simulation_config)
        return

    pred_map = predicted_stiffness_map_pa / 1e3
    true_map = example_stiffness_map_pa / 1e3
    curves_nN = test_collection["measured_force_curves_n"]
    disp_um = test_collection["displacement_curves_m"]
    curve_lengths = test_collection["curve_lengths"]
    retract_starts = test_collection["retract_start_indices"]
    cp_true_map = test_collection["contact_point_map_m"] * 1e9
    cp_pred_map = (
        predicted_contact_point_map_m * 1e9
        if predicted_contact_point_map_m is not None
        else None
    )

    heatmap_fig = go.FigureWidget(data=[go.Heatmap(
        z=pred_map,
        colorscale="magma",
        colorbar={"title": "kPa", "thickness": 18},
        hovertemplate="x=%{x}<br>y=%{y}<br>pred=%{z:.2f} kPa<extra></extra>",
    )])
    heatmap_fig.update_layout(
        title={"text": f"Predicted Stiffness", \
               "y": 0.98, "x": 0.5, "xanchor": "center", \
               "yanchor": "top", "font": {"size": 20}},
        xaxis_title="X",
        yaxis_title="Y",
        width=map_width,
        height=map_height,
        margin=dict(l=55, r=20, t=55, b=55),
        font={"size": 14},
    )
    heatmap_fig.update_yaxes(autorange="reversed", tickfont={"size": 12}, \
                             title_font={"size": 15})
    heatmap_fig.update_xaxes(tickfont={"size": 12}, title_font={"size": 15})

    curve_fig = go.FigureWidget(data=[
    go.Scatter(name="Approach", mode="lines", line={"width": 3}),
    go.Scatter(name="Retract", mode="lines", line={"width": 3}),
    go.Scatter(name="Hertz fit", mode="lines", line={"width": 2, "dash": "dash", "color": "green"}),
])
    curve_fig.update_layout(
        title={"text": "Select a pixel", "y": 0.98, "x": 0.5, \
               "xanchor": "center", "yanchor": "top", "font": {"size": 18}},
        xaxis_title="Displacement [um]",
        yaxis_title="Force [nN]",
        width=curve_width,
        height=curve_height,
        margin=dict(l=60, r=25, t=75, b=55),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, \
                    font={"size": 13}),
        font={"size": 14},
    )
    curve_fig.update_xaxes(tickfont={"size": 12}, title_font={"size": 15})
    curve_fig.update_yaxes(tickfont={"size": 12}, title_font={"size": 15})

    def handle_click(
            trace: go.Scatter, 
            points, 
            selector
        ) -> None:
        """Update the force curve plot based on the clicked pixel in the stiffness map.

        Parameters
        ----------
        trace : go.Scatter
            The scatter trace that was clicked.
        points : go.Points
            The points that were clicked, containing the x and y coordinates.
        selector : go.Selector
            The selector object (not used in this function).

        Returns
        -------
        None
            Updates the force curve plot with the data corresponding to the clicked pixel.

        """
        
        if not points.xs:
            return
        col_index = int(points.xs[0])
        row_index = int(points.ys[0])
        valid_length = int(curve_lengths[row_index, col_index])
        retract_start = int(retract_starts[row_index, col_index])
        x_curve = disp_um[row_index, col_index]
        y_curve = curves_nN[row_index, col_index]
        hertz = approach_force[row_index, col_index] if approach_force is not None else None
        title = (
            f"Pixel ({row_index}, {col_index}) | "
            f"True {true_map[row_index, col_index]:.1f} kPa | "
            f"Predicted {pred_map[row_index, col_index]:.1f} kPa"
        )
        if cp_true_map is not None and cp_pred_map is not None:
            title += (
                f" | Contact point true {cp_true_map[row_index, col_index]:.0f} nm"
                f" predicted {cp_pred_map[row_index, col_index]:.0f} nm"
            )
        with curve_fig.batch_update():
            curve_fig.data[0].x = x_curve[:retract_start] * 1e6
            curve_fig.data[0].y = y_curve[:retract_start] * 1e9
            curve_fig.data[1].x = x_curve[retract_start:valid_length] * 1e6
            curve_fig.data[1].y = y_curve[retract_start:valid_length] * 1e9
            if hertz is not None:
                curve_fig.data[2].x = x_curve[:valid_length] * 1e6
                curve_fig.data[2].y = hertz[:valid_length] * 1e9
            else:
                curve_fig.data[2].x = []
                curve_fig.data[2].y = []
            curve_fig.layout.title.text = title

        if abstract_figure:
            import inspect
            import os
            os.makedirs(save_dir, exist_ok=True)

            # Compute GT Hertz fit from true stiffness and true contact point.
            cp_true_m = cp_true_map[row_index, col_index] / 1e9   # nm → m
            E_true_pa = true_map[row_index, col_index] * 1e3       # kPa → Pa
            disp_m = disp_um[row_index, col_index]                 # meters
            indent_gt = np.maximum(disp_m[:valid_length] - cp_true_m, 0.0)
            cm = contact_model() if inspect.isclass(contact_model) \
                else contact_model
            sc = simulation_config() if inspect.isclass(simulation_config) \
                else simulation_config
            gt_force_n = cm.force(
                cantilever_shape=sc.tip_shape,
                indentation_m=indent_gt,
                stiffness_pa=E_true_pa,
            )

            fig_abs, ax_abs = plt.subplots(figsize=(10, 6))
            ax_abs.plot(
                disp_m[:retract_start] * 1e6,
                y_curve[:retract_start] * 1e9,
            )
            if hertz is not None:
                ax_abs.plot(
                    disp_m[:valid_length] * 1e6,
                    hertz[:valid_length] * 1e9,
                    linewidth=4,
                )
            # ax_abs.plot(
            #     disp_m[:valid_length] * 1e6,
            #     gt_force_n * 1e9,
            # )
            ax_abs.axis("off")
            fig_abs.patch.set_alpha(0)
            ax_abs.patch.set_alpha(0)
            save_path = os.path.join(
                save_dir, f"abstract_{row_index}_{col_index}.png"
            )
            fig_abs.savefig(
                save_path, transparent=True,
                bbox_inches="tight", pad_inches=0,
            )
            plt.close(fig_abs)
            print(f"Saved abstract figure → {save_path}")

    heatmap_fig.data[0].on_click(handle_click)
    display(
        widgets.HBox(
            [heatmap_fig, curve_fig],
            layout=widgets.Layout(width="100%", \
                        justify_content="space-between", align_items="center"),
        )
    )
