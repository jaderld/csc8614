from transformers import GPT2Model
import plotly.express as px
from sklearn.decomposition import PCA

model = GPT2Model.from_pretrained("gpt2")

# 1 : embeddings positionnels
position_embeddings = model.wpe.weight

print("Shape position embeddings:", position_embeddings.size())

print("n_embd:", model.config.n_embd)
print("n_positions:", model.config.n_positions)

# 2 : PCA sur les 50 premières positions
positions_50 = position_embeddings[:50].detach().cpu().numpy()

pca = PCA(n_components=2)
reduced_50 = pca.fit_transform(positions_50)

fig_50 = px.scatter(
    x=reduced_50[:, 0],
    y=reduced_50[:, 1],
    text=[str(i) for i in range(len(reduced_50))],
    color=list(range(len(reduced_50))),
    title="Encodages positionnels GPT-2 (PCA, positions 0–50)",
    labels={"x": "PCA 1", "y": "PCA 2"}
)

fig_50.write_html("positions_50.html")

# 3 : PCA sur les 200 premières positions
positions_200 = position_embeddings[:200].detach().cpu().numpy()

reduced_200 = pca.fit_transform(positions_200)

fig_200 = px.scatter(
    x=reduced_200[:, 0],
    y=reduced_200[:, 1],
    text=[str(i) for i in range(len(reduced_200))],
    color=list(range(len(reduced_200))),
    title="Encodages positionnels GPT-2 (PCA, positions 0–200)",
    labels={"x": "PCA 1", "y": "PCA 2"}
)

fig_200.write_html("positions_200.html")
