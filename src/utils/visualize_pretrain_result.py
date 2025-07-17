import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle

# Data (same as yours)
data_binomial = {
    "en": {"bpc": 1.5919699668884277, "compression_rate": 2.99910624253652, "compression_var": 0.01280173805246407},
    "es": {"bpc": 1.4411402940750122, "compression_rate": 3.6103050265260923, "compression_var": 0.02272494579314865},
    "ru": {"bpc": 0.8874958157539368, "compression_rate": 6.086348303065581, "compression_var": 0.12605270697132628},
    "uk": {"bpc": 0.8964882493019104, "compression_rate": 5.702139985397663, "compression_var": 0.08667069236130527},
    "hi": {"bpc": 0.6909862160682678, "compression_rate": 7.776093708649847, "compression_var": 0.25211693639404903},
    "te": {"bpc": 0.6604868173599243, "compression_rate": 8.237963858149717, "compression_var": 0.2968220933058243}
}
data_FlexiTokens = {
    "en": {"bpc": 1.5927218198776245, "compression_rate": 3.2176143357231077, "compression_var": 0.014196921791226232},
    "es": {"bpc": 1.443312406539917, "compression_rate": 3.9123372039487285, "compression_var": 0.022386434653233134},
    "ru": {"bpc": 0.8851881623268127, "compression_rate": 6.545856525477241, "compression_var": 0.08634897731134997},
    "uk": {"bpc": 0.8944931030273438, "compression_rate": 6.167112109664079, "compression_var": 0.07627719453769677},
    "hi": {"bpc": 0.694412350654602, "compression_rate": 8.605481567143613, "compression_var": 0.2430539701556464},
    "te": {"bpc": 0.6618195176124573, "compression_rate": 9.033035374346749, "compression_var": 0.2568919762457847}
}
data_FlexiTokens_1B = {
    "en": {"bpc": 1.3782336711883545, "compression_rate": 3.5098588852588732, "compression_var": 0.014741941293683777},
    "es": {"bpc": 1.2486897706985474, "compression_rate": 4.1190475737811285, "compression_var": 0.013959593007643118},
    "ru": {"bpc": 0.7313501834869385, "compression_rate": 6.791618883197738, "compression_var": 2.981946354273422},
    "uk": {"bpc": 0.7313182950019836, "compression_rate": 6.408126632012764, "compression_var": 0.03219568561425283},
    "hi": {"bpc": 0.5611947178840637, "compression_rate": 8.970996926778144, "compression_var": 0.11264009389728927},
    "te": {"bpc": 0.5246986746788025, "compression_rate": 9.37368181063805, "compression_var": 0.08566487156314614}
}

# Build DataFrame with all three datasets
langs = ["en", "es", "ru", "uk", "hi", "te"]
records = []
for lang in langs:
    # Calculate standard deviation as sqrt of variance
    binomial_std = np.sqrt(data_binomial[lang]["compression_var"])
    FlexiTokens_std = np.sqrt(data_FlexiTokens[lang]["compression_var"])
    FlexiTokens_1B_std = np.sqrt(data_FlexiTokens_1B[lang]["compression_var"])
    records.append({
        "language": lang,
        "Bits per Character (Binomial)": data_binomial[lang]["bpc"],
        "Bits per Character (FlexiTokens)": data_FlexiTokens[lang]["bpc"],
        "Bits per Character (FlexiTokens 1B)": data_FlexiTokens_1B[lang]["bpc"],
        "Compression Rate (Binomial)": data_binomial[lang]["compression_rate"],
        "Compression Rate (FlexiTokens)": data_FlexiTokens[lang]["compression_rate"],
        "Compression Rate (FlexiTokens 1B)": data_FlexiTokens_1B[lang]["compression_rate"],
        "Compression StdDev (Binomial)": binomial_std,
        "Compression StdDev (FlexiTokens)": FlexiTokens_std,
        "Compression StdDev (FlexiTokens 1B)": FlexiTokens_1B_std,
    })
df_all = pd.DataFrame.from_records(records)

# Melt and label for BPC and Compression Rate
def melt_and_label(df, cols, value_name):
    melted = df.melt(id_vars="language", value_vars=cols, var_name="Setup", value_name=value_name)
    melted["Setup"] = melted["Setup"].apply(lambda x: "Binomial" if "Binomial" in x else
                                           ("FlexiTokens 1B" if "FlexiTokens 1B" in x else "FlexiTokens"))
    return melted

bpc_plot = melt_and_label(df_all,
                          ["Bits per Character (Binomial)", "Bits per Character (FlexiTokens)", "Bits per Character (FlexiTokens 1B)"],
                          "Bits per Character")

# For compression rate, we need to include stddev
compr_plot = pd.DataFrame()
for lang in langs:
    # Binomial
    compr_plot = pd.concat([compr_plot, pd.DataFrame({
        "language": [lang],
        "Setup": ["Binomial"],
        "Compression Rate": [df_all.loc[df_all["language"] == lang, "Compression Rate (Binomial)"].values[0]],
        "StdDev": [df_all.loc[df_all["language"] == lang, "Compression StdDev (Binomial)"].values[0]]
    })], ignore_index=True)
    # FlexiTokens
    compr_plot = pd.concat([compr_plot, pd.DataFrame({
        "language": [lang],
        "Setup": ["FlexiTokens"],
        "Compression Rate": [df_all.loc[df_all["language"] == lang, "Compression Rate (FlexiTokens)"].values[0]],
        "StdDev": [df_all.loc[df_all["language"] == lang, "Compression StdDev (FlexiTokens)"].values[0]]
    })], ignore_index=True)
    # FlexiTokens 1B
    compr_plot = pd.concat([compr_plot, pd.DataFrame({
        "language": [lang],
        "Setup": ["FlexiTokens 1B"],
        "Compression Rate": [df_all.loc[df_all["language"] == lang, "Compression Rate (FlexiTokens 1B)"].values[0]],
        "StdDev": [df_all.loc[df_all["language"] == lang, "Compression StdDev (FlexiTokens 1B)"].values[0]]
    })], ignore_index=True)

# Define bar annotation function
def annotate_bars(ax, fmt="{:.3f}"):
    # Only apply to bar containers, not error bar or other containers
    for container in ax.containers:
        if isinstance(container, matplotlib.container.BarContainer):
            # Extract values from the rectangles/bars
            values = [rect.get_height() for rect in container]
            labels = [fmt.format(v) for v in values]
            ax.bar_label(container, labels=labels, label_type="edge", padding=10, fontsize=16, rotation=62)

# Use your palette with an additional shade of blue for FlexiTokens 1B
default_palette = ["#D8EEFF", "#0073CF", "#1A5490"]  # Added darker blue for FlexiTokens 1B

# Plot 1x2 with headroom
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Plot 1: Bits Per Character
sns.barplot(data=bpc_plot, x="language", y="Bits per Character", hue="Setup", ax=axs[0],
            palette=default_palette, edgecolor='black', linewidth=0.3)
axs[0].set_title("Bits Per Character (↓ Better)", fontsize=14)
axs[0].set_ylim(0, max(bpc_plot["Bits per Character"]) * 1.25)
axs[0].tick_params(axis='both', labelsize=14)
axs[0].set_xlabel("Language", fontsize=14)
axs[0].set_ylabel("Bits per Character", fontsize=14)
axs[0].grid(True, axis='both', linestyle='--', alpha=0.7)
annotate_bars(axs[0])

# Plot 2: Compression Rate with Error Bars
# First create the base barplot
bars = sns.barplot(data=compr_plot, x="language", y="Compression Rate", hue="Setup", ax=axs[1],
                  palette=default_palette, edgecolor='black', linewidth=0.3)

# Extract bar positions for error bars
bar_positions = []
bar_heights = []
for container in axs[1].containers:
    for bar in container.patches:
        bar_positions.append(bar.get_x() + bar.get_width()/2)
        bar_heights.append(bar.get_height())

# Add error bars manually
binomial_bars = bar_positions[:6]  # First 6 bars are binomial
FlexiTokens_bars = bar_positions[6:12]  # Next 6 bars are FlexiTokens
FlexiTokens_1B_bars = bar_positions[12:]  # Last 6 bars are FlexiTokens 1B

# Add error bars for Binomial
binomial_stddev = compr_plot[compr_plot["Setup"] == "Binomial"]["StdDev"].values
for pos, height, std in zip(binomial_bars, compr_plot[compr_plot["Setup"] == "Binomial"]["Compression Rate"].values, binomial_stddev):
    axs[1].errorbar(pos, height, yerr=std, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)

# Add error bars for FlexiTokens
FlexiTokens_stddev = compr_plot[compr_plot["Setup"] == "FlexiTokens"]["StdDev"].values
for pos, height, std in zip(FlexiTokens_bars, compr_plot[compr_plot["Setup"] == "FlexiTokens"]["Compression Rate"].values, FlexiTokens_stddev):
    axs[1].errorbar(pos, height, yerr=std, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)

# Add error bars for FlexiTokens 1B
FlexiTokens_1B_stddev = compr_plot[compr_plot["Setup"] == "FlexiTokens 1B"]["StdDev"].values
for pos, height, std in zip(FlexiTokens_1B_bars, compr_plot[compr_plot["Setup"] == "FlexiTokens 1B"]["Compression Rate"].values, FlexiTokens_1B_stddev):
    axs[1].errorbar(pos, height, yerr=std, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)

axs[1].set_title("Compression Rate with StdDev (↑ Better)", fontsize=14)
axs[1].set_ylim(0, max(compr_plot["Compression Rate"] + compr_plot["StdDev"]) * 1.2)
axs[1].tick_params(axis='both', labelsize=14)
axs[1].set_xlabel("Language", fontsize=14)
axs[1].set_ylabel("Compression Rate", fontsize=14)
axs[1].grid(True, axis='both', linestyle='--', alpha=0.7)
annotate_bars(axs[1])

# Legend tweaks
for ax in axs:
    ax.legend(title="Loss Type", loc="best", fontsize=11, title_fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('model_performance_fineweb_test_with_errorbars.png', dpi=300, bbox_inches='tight')
plt.show()