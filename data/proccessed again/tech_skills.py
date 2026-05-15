"""
Curated technical skill vocabulary to layer on top of the O*NET skills.

Two structures are exported:
    TECH_SKILLS : set[str]
        Canonical lowercase skill strings. Multi-word skills use single spaces.

    TECH_ALIASES : dict[str, str]
        Canonical-form -> safe single-token replacement, used for skills
        whose canonical form contains characters that don't tokenize cleanly
        (e.g. "c++", "c#", ".net"). The pipeline substitutes these BEFORE
        tokenization and reverses the substitution when surfacing extracted
        skills back to the user.

Add or remove entries here freely; the pipeline picks up changes automatically.
"""

# ---------------------------------------------------------------------------
# Canonical multi-word and simple skills
# ---------------------------------------------------------------------------
_LANGUAGES = {
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "perl", "r", "matlab", "sql",
    "bash", "shell scripting", "powershell", "html", "css", "sass", "less",
    "objective c", "dart", "lua", "groovy", "haskell", "elixir", "clojure",
    "fortran", "cobol", "vba", "assembly",
}

_WEB_FRAMEWORKS = {
    "react", "react native", "angular", "vue", "svelte", "next js", "nuxt js",
    "node js", "express", "express js", "django", "flask", "fastapi", "spring",
    "spring boot", "ruby on rails", "rails", "laravel", "symfony", "asp net",
    "blazor", "ember", "backbone", "jquery", "bootstrap", "tailwind",
    "tailwind css", "material ui", "redux", "graphql", "rest api", "grpc",
    "websocket", "soap", "openapi", "swagger",
}

_DATA_ML = {
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "data science",
    "data engineering", "data analytics", "data analysis", "data mining",
    "feature engineering", "model deployment", "mlops", "a/b testing",
    "time series", "recommender systems", "anomaly detection",
    "tensorflow", "pytorch", "keras", "scikit learn", "sklearn", "xgboost",
    "lightgbm", "catboost", "pandas", "numpy", "scipy", "statsmodels",
    "matplotlib", "seaborn", "plotly", "bokeh", "dash", "streamlit",
    "hugging face", "transformers", "langchain", "llamaindex", "openai",
    "spacy", "nltk", "gensim", "opencv", "pillow", "fastai", "jax",
    "mlflow", "kubeflow", "weights and biases", "wandb", "ray", "dvc",
    "prefect", "dagster", "airflow",
}

_BIG_DATA = {
    "spark", "pyspark", "hadoop", "hive", "presto", "trino", "kafka",
    "flink", "storm", "beam", "databricks", "snowflake", "redshift",
    "bigquery", "athena", "glue", "emr", "dataproc", "dbt", "fivetran",
    "stitch", "segment",
}

_DATABASES = {
    "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
    "dynamodb", "couchbase", "couchdb", "oracle", "sql server", "sqlite",
    "mariadb", "elasticsearch", "opensearch", "solr", "neo4j", "arangodb",
    "influxdb", "timescaledb", "clickhouse", "duckdb", "firestore",
    "supabase", "cockroachdb",
}

_CLOUD = {
    "aws", "amazon web services", "azure", "microsoft azure", "gcp",
    "google cloud", "google cloud platform", "ibm cloud", "oracle cloud",
    "alibaba cloud", "digitalocean", "heroku", "vercel", "netlify",
    "cloudflare",
    "ec2", "s3", "lambda", "ecs", "eks", "fargate", "rds", "cloudwatch",
    "cloudformation", "iam", "vpc", "route53", "sqs", "sns",
    "gke", "cloud run", "app engine", "cloud functions", "pub sub",
    "cloud sql", "spanner", "bigtable", "vertex ai", "sagemaker",
}

_DEVOPS = {
    "docker", "kubernetes", "k8s", "helm", "istio", "terraform", "pulumi",
    "ansible", "puppet", "chef", "salt", "jenkins", "github actions",
    "gitlab ci", "circleci", "travis ci", "bamboo", "argo cd", "argocd",
    "spinnaker", "prometheus", "grafana", "datadog", "new relic",
    "splunk", "elk", "logstash", "kibana", "fluentd", "fluent bit",
    "opentelemetry", "jaeger", "sentry", "pagerduty", "opsgenie",
    "vagrant", "packer", "consul", "vault", "nomad", "linkerd",
}

_TOOLS = {
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello",
    "asana", "notion", "slack", "figma", "sketch", "postman", "insomnia",
    "vs code", "intellij", "pycharm", "eclipse", "jupyter", "colab",
    "tableau", "power bi", "looker", "qlik", "metabase", "superset",
    "excel", "google sheets",
}

_MOBILE = {
    "ios", "android", "flutter", "xamarin", "ionic", "cordova", "swiftui",
    "jetpack compose", "kotlin multiplatform",
}

_CONCEPTS = {
    "microservices", "monolith", "serverless", "event driven", "domain driven design",
    "ddd", "tdd", "bdd", "ci cd", "agile", "scrum", "kanban", "waterfall",
    "etl", "elt", "data warehouse", "data lake", "data lakehouse", "lakehouse",
    "oop", "functional programming", "design patterns", "rest", "soa",
    "oauth", "jwt", "saml", "sso", "encryption", "tls", "ssl", "pki",
    "linux", "unix", "windows server", "macos", "networking", "tcp ip",
    "load balancing", "high availability", "fault tolerance",
    "distributed systems", "concurrency", "multithreading", "caching",
    "message queue", "service mesh",
}

# Skills whose canonical form has characters that don't tokenize cleanly.
# Mapped to single-token aliases that survive tokenization. The pipeline
# substitutes the LHS with the RHS before cleaning/tokenization, and reverses
# it when surfacing extracted_skills back to the user.
TECH_ALIASES: dict[str, str] = {
    "c++": "cplusplus",
    "c#": "csharp",
    "f#": "fsharp",
    "objective-c": "objectivec",
    ".net": "dotnet",
    "asp.net": "aspdotnet",
    "node.js": "nodejs",
    "next.js": "nextjs",
    "nuxt.js": "nuxtjs",
    "vue.js": "vuejs",
    "ember.js": "emberjs",
    "express.js": "expressjs",
    "d3.js": "d3js",
    "three.js": "threejs",
    "ci/cd": "cicd",
    "tcp/ip": "tcpip",
    "a/b testing": "abtesting",
}

TECH_SKILLS: set[str] = (
    _LANGUAGES
    | _WEB_FRAMEWORKS
    | _DATA_ML
    | _BIG_DATA
    | _DATABASES
    | _CLOUD
    | _DEVOPS
    | _TOOLS
    | _MOBILE
    | _CONCEPTS
    | set(TECH_ALIASES.values())   # canonical aliased forms
)


# ---------------------------------------------------------------------------
# Ambiguous tokens: skills whose name collides with common English words.
# Adding them to the vocabulary alone produces false positives ("rest of the
# team", "30 ML of solution", "Brian Smith"). The preprocessing pipeline
# applies a raw-text capitalization / context check before counting them as
# real skill matches.
#
# SHORT  : 1-2 char tokens. Need BOTH capitalization AND a strong context cue
#          in the raw text window. Strict because false positives explode.
# LONG   : 3+ char tokens. Capitalization OR any context cue is sufficient.
# ---------------------------------------------------------------------------
AMBIGUOUS_TOKENS_SHORT: set[str] = {
    # original (carried over from the v2 build_final_dataset.py logic)
    "r", "go",
    # alias targets that survived the prompt review
    "ml", "dl", "cv", "tf", "bi", "ux", "ui",
}

AMBIGUOUS_TOKENS_LONG: set[str] = {
    # original collisions documented in PIPELINE_LOG.md
    "less", "express", "swift", "storm", "segment", "sketch",
    # 3-letter ML/AI abbreviations the user wants to keep
    "cnn", "rnn", "svm", "knn", "pca", "eda", "rag",
    # business systems abbreviations
    "crm", "erp",
}

AMBIGUOUS_TOKENS: set[str] = AMBIGUOUS_TOKENS_SHORT | AMBIGUOUS_TOKENS_LONG


# Tech-context cues considered when judging an ambiguous match. The
# STRONG cue list is deliberately small so SHORT tokens don't get free
# passes from generic words like "with" or "from".
WEAK_CONTEXT_CUES: set[str] = {
    "language", "programming", "develop", "framework", "library", "package",
    "stack", "code", "coding", "scripting", "lang", "skills", "tools",
    "experience", "proficient", "knowledge", "using", "with",
}

STRONG_CONTEXT_CUES: set[str] = {
    "programming", "framework", "library", "developer", "engineer",
    "skills", "tools", "experience", "language",
}
