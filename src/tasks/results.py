import lib.util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def toxicity_barplot(df: pd.DataFrame, ax):
    sns.barplot(
        data=df,
        y="conv_variant",
        x="toxicity",
        hue="annotator_prompt",
        estimator=np.mean,
        ax=ax,
    )
    ax.axvline(x=3, color="r")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim(0, 5)
    ax.legend(title="Annotator Demographic", fontsize="6", title_fontsize="6.5")


def pvalue_heatmap(
    value_df: pd.DataFrame,
    pvalue_df: pd.DataFrame,
    show_labels: bool = False,
    correlation_title: str = "",
    xlabel_text: str = "",
    filename: str | None = None,
    output_dir: str = "."
) -> None:
    """Generate a heatmap with p-values and corresponding asterisks

    :param value_df: DataFrame with correlation or other values
    :param pvalue_df: DataFrame with p-values corresponding to value_df
    :param show_labels: Boolean indicating whether to show labels on the heatmap
    :param correlation_title: Title for the heatmap
    :param filename: Optional filename to save the plot
    """
    # Format the value_df with asterisks based on pvalue_df
    formatted_values = _format_with_asterisks(value_df, pvalue_df)

    # Define tick labels
    ticklabels = value_df.columns if show_labels else "auto"

    # Create the heatmap
    sns.heatmap(
        np.tril(value_df),
        annot=np.tril(formatted_values),
        fmt="",  # This allows us to use strings with asterisks
        cmap="icefire",
        mask=_upper_tri_masking(value_df),
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"label": "Mean Toxicity Difference"},
        annot_kws={"fontsize":8}
    )

    plt.title(correlation_title)
    plt.xlabel(xlabel_text)

    # Save the plot if a filename is provided
    if filename is not None:
        lib.util.save_plot(filename=filename, dir_name=output_dir)

    # Show the plot
    plt.show()


# code from https://stackoverflow.com/questions/47314754/how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy
def _upper_tri_masking(array: np.array) -> np.array:
    """Generate a mask for the upper triangular of a NxN matrix, without the main diagonal

    :param array: the NxN matrix
    :type array: np.array
    :return: the mask
    :rtype: np.array
    """
    m = array.shape[0]
    r = np.arange(m)
    mask = r[:, None] <= r
    return mask


def _format_with_asterisks(
    value_df: pd.DataFrame, pvalue_df: pd.DataFrame
) -> pd.DataFrame:
    """Format the values in the value_df with asterisks based on p-value significance levels

    :param value_df: DataFrame containing the values to display
    :param pvalue_df: DataFrame containing the p-values
    :return: DataFrame with values formatted with asterisks
    """
    formatted_df = value_df.copy().astype(str)
    for i in range(value_df.shape[0]):
        for j in range(value_df.shape[1]):
            value = value_df.iloc[i, j]
            pvalue = pvalue_df.iloc[i, j]
            if pd.notnull(pvalue):  # Only apply formatting if pvalue is not NaN
                if pvalue < 0.001:
                    num_asterisks = 3
                elif pvalue < 0.01:
                    num_asterisks = 2
                elif pvalue < 0.05:
                    num_asterisks = 1
                else:
                    num_asterisks = 0
            else: #if NaN
                num_asterisks = 0

            formatted_df.iloc[i, j] = f"{value:.3f}{num_asterisks * '*'}"

    return formatted_df
