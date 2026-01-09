# CSC8614 — TP1

**Nom Prénom** : ROLAND Jade  

## Environnement & reproductibilité

- **OS** : Windows 10  
- **Python** : 3.10  
- **Bibliothèques principales** :
  - torch
  - transformers
  - scikit-learn
  - plotly

### Installation / activation de l’environnement

```bash
python -m venv venv
source venv/Scripts/activate
cd ./TP1
pip install -r requirements.txt


## Exercice 1 - Découverte du tokenizer GPT-2

### 1.a
Tokens:
['Art', 'ificial', 'Ġintelligence', 'Ġis', 'Ġmet', 'amorph', 'osing', 'Ġthe', 'Ġworld', '!']

Dans le tokenizer GPT-2, le symbole Ġ indique qu’un token est précédé d’un espace dans le texte original. GPT-2 ne traite pas explicitement les espaces comme des tokens séparés, mais les encode dans les sous-mots eux-mêmes. Cela permet au modèle de conserver l’information de segmentation entre les mots tout en utilisant un vocabulaire de sous-tokens. Ce mécanisme est typique des tokenizers BPE utilisés par les modèles de type GPT.

### 1.b
| Token (décodé) | ID | Remarque |
|---------------|----|----------|
| `Art` | 8001 | Début de phrase, pas d’espace |
| `ificial` | 9542 | Suite du mot précédent |
| ` intelligence` | 4430 | Espace encodé (`Ġ`) + mot |
| ` is` | 318 | Espace encodé (`Ġ`) |
| ` met` | 1138 | Début d’un mot long, avec espace |
| `amorph` | 37670 | Sous-mot intermédiaire (BPE) |
| `osing` | 2752 | Fin du mot *metamorphosing* |
| ` the` | 262 | Mot fréquent avec espace |
| ` world` | 995 | Mot fréquent avec espace |
| `!` | 0 | Ponctuation, token séparé |

Les tokens sont les unités textuelles produites par la tokenisation. Les token IDs sont leurs représentations numériques associées. Le modèle de langage ne manipule que les IDs.

### 1.c
Les mots fréquents comme intelligence ou world sont souvent encodés en un seul token.
Les mots longs ou rares comme metamorphosing sont découpés en plusieurs sous-tokens.
La ponctuation (!, .) est généralement séparée en tokens distincts.
Les espaces ne sont pas des tokens indépendants mais intégrés via le préfixe Ġ.

Ces observations illustrent le principe du Byte Pair Encoding, qui privilégie la réutilisation de fragments fréquents afin de limiter la taille du vocabulaire tout en couvrant un grand nombre de mots.

### 1.d
Tokens phrase 2:
['G', 'PT', 'Ġmodels', 'Ġuse', 'ĠB', 'PE', 'Ġtoken', 'ization', 'Ġto', 'Ġprocess', 'Ġunusual', 'Ġwords', 'Ġlike', 'Ġant', 'idis', 'establishment', 'arian', 'ism', '.']

Sous-tokens du mot 'antidisestablishmentarianism':
['Ġant', 'idis', 'establishment', 'arian', 'ism']
Nombre de sous-tokens: 5

Le mot antidisestablishmentarianism, très long et rare, est découpé en de nombreux sous-tokens correspondant à des fragments plus fréquents comme anti, dis, establish, ment, etc. Cela permet au modèle de traiter des mots inconnus ou peu fréquents en les recomposant à partir d’unités connues.


## Exercice 2 - Analyse des embeddings positionnels dans GPT-2

### 2.a
`Shape position embeddings: torch.Size([1024, 768])`

La matrice des encodages positionnels a une shape (1024, 768).
Chaque ligne correspond à une position possible dans la séquence, et chaque colonne à une dimension de l’espace d’embedding. Pour GPT-2, le modèle peut représenter jusqu’à n_positions = 1024 positions distinctes, chacune encodée dans un espace de dimension n_embd = 768.

Le paramètre n_positions correspond à la longueur maximale du contexte que le modèle peut traiter. GPT-2 étant un modèle de langage causal, il ne peut conditionner sa prédiction que sur les n_positions derniers tokens. Toute séquence plus longue doit être tronquée ou découpée, ce qui limite la mémoire contextuelle du modèle.

### 2.b
capture d’écran du HTML positions_50.html

La projection PCA des encodages positionnels pour les positions 0 à 50 montre une trajectoire relativement continue dans l’espace 2D. Les points correspondant aux positions successives sont proches les uns des autres, ce qui suggère une variation progressive des encodages avec la position. On observe peu de regroupements discrets, mais plutôt une structure lisse, indiquant que le modèle encode la notion de proximité positionnelle. Cela reflète le fait que des positions proches doivent produire des effets similaires dans le mécanisme d’attention.
Les encodages positionnels vivent dans un espace de grande dimension (768). La PCA permet de projeter ces vecteurs en 2D tout en conservant au maximum la variance, rendant possible une visualisation interprétable de leur structure globale.

### 2.c
capture d’écran de positions_200.html

Lorsque l’on étend la visualisation aux positions 0 à 200, la structure devient plus étalée et parfois moins lisible localement. Les premières positions restent relativement bien organisées, mais les positions plus éloignées occupent des régions plus larges de l’espace projeté. Cela suggère que le modèle apprend à différencier fortement les positions lointaines afin d’éviter les confusions dans de longues séquences. Une hypothèse est que GPT-2 encode la position de manière non périodique et non strictement linéaire, ce qui permet de préserver l’ordre tout en maximisant la capacité de représentation sur de longues distances.


## Exercice 3 - Probabilités et génération de texte avec GPT-2

### 3.a
Probabilités conditionnelles par token :
1 'ificial' 1.920e-05
2 ' intelligence' 1.505e-01
3 ' is' 1.955e-01
4 ' fascinating' 6.504e-04
5 '.' 1.773e-01

GPT-2 est un modèle de langage causal : à la position t, il prédit le token t+1. Les logits à l’indice t−1 correspondent donc à la distribution de probabilité du token observé à la position t. C’est pourquoi la probabilité de token_t est lue dans logits[t-1]. Le premier token n’a pas de probabilité conditionnelle car aucun contexte ne le précède.

### 3.b
Log-proba totale: -23.454901337623596
Avg negative log-proba: 4.690980267524719
Perplexité: 108.95993736147643

La perplexité mesure à quel point le modèle est surpris par une séquence. Intuitivement, elle correspond au nombre effectif de choix possibles que le modèle considère à chaque étape. Une faible perplexité indique que la phrase est bien alignée avec les régularités apprises par le modèle, tandis qu’une perplexité élevée indique une phrase improbable ou peu naturelle. La perplexité est une moyenne exponentielle de la log-perte par token et permet de comparer des phrases de longueurs différentes.

### 3.c
Phrase: Artificial intelligence is fascinating.
total_logp: -23.454901337623596
avg_neg_logp: 4.690980267524719
perplexity: 108.95993736147643

Phrase: Artificial fascinating intelligence is.
total_logp: -42.164613246917725
avg_neg_logp: 8.432922649383546
perplexity: 4595.91272469554

La phrase grammaticalement correcte présente une perplexité significativement plus faible que la phrase avec un ordre des mots incorrect. GPT-2 a appris des régularités syntaxiques fortes à partir de grandes quantités de texte naturel. Une permutation non naturelle viole ces régularités et entraîne une baisse des probabilités conditionnelles pour plusieurs tokens, ce qui augmente fortement la perplexité globale.

### 3.d
Phrase: L'intelligence artificielle est fascinante.
total_logp: -59.481457352638245
avg_neg_logp: 5.948145735263824
perplexity: 383.04241809863305

GPT-2 étant majoritairement entraîné sur des données en anglais, les phrases françaises ont une perplexité plus élevée. Les tokens français sont moins fréquents, souvent découpés en sous-tokens inhabituels, et leurs cooccurrences sont moins bien modélisées, ce qui augmente la surprise du modèle.

### 3.e
Top-10 tokens probables après le préfixe :
' a' 1.204e-01
' the' 5.254e-02
' not' 4.324e-02
' an' 3.092e-02
' now' 2.062e-02
' one' 1.890e-02
' also' 1.880e-02
' already' 1.716e-02
' becoming' 1.606e-02
' just' 1.422e-02

Les tokens proposés sont sémantiquement et grammaticalement plausibles. On observe la présence de tokens commençant par un espace, ce qui reflète le fonctionnement du tokenizer GPT-2. La distribution est concentrée sur quelques mots fréquents, indiquant une forte confiance du modèle dans la continuation.


## Exercice 4 - Exploration des méthodes de génération avec GPT-2

### 4.a
Seed utilisé : 42
Fixer le seed garantit que les résultats sont reproductibles. Les modèles génératifs utilisant sampling ou beam search contiennent des éléments aléatoires, et sans seed, les sorties changeraient à chaque exécution.

### 4.b
Greedy decoding: The future of artificial intelligence is uncertain. "We're not sure what the future will look like," said Dr. Michael S. Schoenfeld, a professor of computer science at the University of California, Berkeley.

On observe que les sorties sont toujours identiques pour un même seed. Le texte est cohérent et grammaticalement correct, et il n'y a pas de diversité : le modèle choisit toujours le token le plus probable à chaque étape.

### 4.c
SEED 1 :
The future of artificial intelligence is up in the air, and the future of artificial intelligence is now about to change. For now, we're just waiting for the technology to be perfected so that we can take it to the next level.

SEED 2 :
The future of artificial intelligence is not clear, but that could change. The early progress of AI has been largely due to the ability to do some things fairly quickly, like calculate things, but the future is not clear. The early progress of AI has

En comparant avec le greedy decoding, on remarque que les phrases générées par sampling sont différentes à chaque seed, alors que le greedy produit toujours la même sortie : « The future of artificial intelligence is uncertain. » Les phrases sampling restent globalement cohérentes avec le prompt mais peuvent inclure des phrases incomplètes. Sampling peut aussi répéter certains segments (“the future of artificial intelligence…”) plus qu’en greedy, mais cela dépend de la combinaison de top-k, top-p et seed.

Température (0.7 ici) : Contrôle la softmax sur les logits avant l’échantillonnage.
Temp < 1 : distribution plus concentrée sur les tokens les plus probables (moins de diversité).
Temp > 1 : distribution plus aplatie, plus de diversité mais risque de sorties incohérentes.
Top-k (50) : Restreint l’échantillonnage aux 50 tokens les plus probables à chaque étape. Cela empêche de choisir des tokens très improbables qui ruineraient la cohérence.
Top-p (0.95) : Échantillonne parmi les tokens dont la somme est de probabilité ≤ 0.95, garantissant que les tokens rares mais cohérents sont encore possibles.

### 4.d
Sans pénalité (seed=1, temp=0.7)
`The future of artificial intelligence is up in the air, and the future of artificial intelligence is now about to change. For now, we're just waiting for the technology to be perfected so that we can take it to the next level.`

Avec pénalité de répétition (repetition_penalty=2.0, seed=1, temp=0.7)
`The future of artificial intelligence is up in the air, and it may not be as interesting or useful to us humans. But we're going down a path where our ability for thinking about things could become less important than ever before."`

Avec repetition_penalty=2.0, les répétitions directes de segments (“the future of artificial intelligence”) disparaissent. Le texte reste cohérent mais devient parfois un peu plus verbose. Certaines phrases deviennent légèrement plus complexes pour éviter de répéter les tokens, ce qui peut donner des constructions un peu moins naturelles.

### 4.e
Température très basse (0.1)
`The future of artificial intelligence is uncertain. But the future of artificial intelligence is not.`
`The future of artificial intelligence is not.`
`The future of artificial intelligence is not.`

Température très élevée (2.0)
`The future of artificial intelligence is up in the air again in 2014 as Google unveils its new platform called MachineStory-AI called Watson from the Stanford Institute for Artificial Intelligence (SetBorg). For IBM and for everyone trying to get their heads in`

Lorsque l’on utilise une température très basse, par exemple 0.1, le texte généré est très cohérent. Le modèle choisit quasiment toujours le token le plus probable à chaque étape, ce qui produit des phrases grammaticalement correctes, mais extrêmement répétitives et peu créatives.

À l’inverse, une température très élevée produit un texte beaucoup plus divers et créatif. Le modèle explore des tokens rares et des constructions plus éloignées prompt initial. Cela donne des phrases parfois incohérentes ou improbables.

Ainsi, le choix de la température constitue un compromis entre cohérence et diversité. Une température basse favorise un texte stable et grammatical, adapté à un usage informatif, tandis qu’une température élevée permet d’obtenir des textes plus créatifs, avec néanmoins un risque accru d’incohérences ou d’erreurs factuelles.

### 4.f
Beam search num_beams=5:
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The future of artificial intelligence is in the hands of the next generation of scientists and engineers.

The future of artificial intelligence is in the hands of the next generation of scientists and engineers.

The future of artificial intelligence is in the hands of
Temps approximatif: 3.88s

Comparé au décodage glouton, le beam search produit un texte plus probable globalement. Le modèle prend en compte plusieurs séquences candidates simultanément et choisit la suite qui maximise la probabilité totale, ce qui tend à éviter certaines erreurs locales que greedy pourrait commettre. Cela se traduit par un texte grammaticalement correct, cohérent et plausible, mais un peu plus générique : on remarque moins de variations que lors du sampling, et le texte est moins diversifié.

### 4.g
Beam search num_beams=10:
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The future of artificial intelligence is in the hands of the next generation of scientists and engineers.

The future of artificial intelligence is in the hands of the next generation of scientists and engineers.

The future of artificial intelligence is in the hands of
Temps approximatif: 5.51s
------------------------------------------------------------
Beam search num_beams=20:
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The future of artificial intelligence is in the hands of the next generation of scientists and engineers.

The future of artificial intelligence is in the hands of the next generation of scientists and engineers.

The future of artificial intelligence is in the hands of
Temps approximatif: 7.69s

En augmentant le nombre de beams à 10 puis 20, le temps de génération a augmenté à 5,51s pour 10 beams et 7.69s pour 20 beams. L’augmentation du temps est due à la complexité combinatoire : à chaque pas de génération, le modèle doit maintenir et scorer plusieurs séquences candidates simultanément, explorer davantage de chemins et recalculer les probabilités pour chacune, ce qui multiplie le coût computationnel.