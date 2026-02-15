"""
ML Command
==========

Run the SwarmML pipeline for automated machine learning.

Usage:
    /ml titanic              # Load seaborn titanic and run ML
    /ml path/to/data.csv     # Load CSV and run ML
    /ml --query "SELECT * FROM users" --connection pg  # Load from database
    /ml --help               # Show help

Supported Datasets:
    Seaborn: titanic, iris, tips, penguins, diamonds, mpg
    Sklearn: breast_cancer, wine, digits, california, diabetes
    Files: .csv, .parquet, .json
    Database: PostgreSQL, MySQL, SQLite, etc. (via ConnectorX)

MLflow Integration:
    /ml titanic --mlflow                    # Enable MLflow tracking
    /ml titanic --mlflow --experiment myexp # Custom experiment name
"""

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class MLCommand(BaseCommand):
    """Run SwarmML pipeline for automated machine learning."""

    name = "ml"
    aliases = ["automl", "swarmml"]
    description = "Run SwarmML pipeline - world-class AutoML with LLM feedback"
    usage = "/ml <dataset> [--target <column>] [--context <business_context>] [--iterations <n>] [--mlflow] [--query <sql>] [--connection <name>]"
    category = "ml"

    # Supported datasets
    SEABORN_DATASETS = ["titanic", "iris", "tips", "penguins", "diamonds", "mpg"]
    SKLEARN_DATASETS = ["breast_cancer", "wine", "digits", "california", "diabetes"]

    # Database connection aliases
    DB_ALIASES = {
        "pg": {"db_type": "postgresql", "host": "localhost", "port": 5432},
        "mysql": {"db_type": "mysql", "host": "localhost", "port": 3306},
        "sqlite": {"db_type": "sqlite"},
    }

    # MLflow state file
    MLFLOW_STATE_FILE = Path.home() / ".jotty" / "mlflow_state.json"

    # Dataset leaderboard - best known scores
    DATASET_LEADERBOARD = {
        "titanic": {
            "type": "classification",
            "target": "survived",
            "baseline": 0.78,
            "best": 0.87,
            "samples": 891,
        },
        "iris": {
            "type": "classification",
            "target": "species",
            "baseline": 0.93,
            "best": 0.98,
            "samples": 150,
        },
        "breast_cancer": {
            "type": "classification",
            "target": "target",
            "baseline": 0.94,
            "best": 0.99,
            "samples": 569,
        },
        "wine": {
            "type": "classification",
            "target": "target",
            "baseline": 0.95,
            "best": 0.99,
            "samples": 178,
        },
        "digits": {
            "type": "classification",
            "target": "target",
            "baseline": 0.95,
            "best": 0.99,
            "samples": 1797,
        },
        "penguins": {
            "type": "classification",
            "target": "species",
            "baseline": 0.95,
            "best": 0.99,
            "samples": 344,
        },
        "tips": {
            "type": "regression",
            "target": "tip",
            "baseline": 0.40,
            "best": 0.65,
            "samples": 244,
        },
        "diamonds": {
            "type": "regression",
            "target": "price",
            "baseline": 0.90,
            "best": 0.98,
            "samples": 53940,
        },
        "mpg": {
            "type": "regression",
            "target": "mpg",
            "baseline": 0.75,
            "best": 0.92,
            "samples": 398,
        },
        "california": {
            "type": "regression",
            "target": "target",
            "baseline": 0.60,
            "best": 0.85,
            "samples": 20640,
        },
        "diabetes": {
            "type": "regression",
            "target": "target",
            "baseline": 0.40,
            "best": 0.55,
            "samples": 442,
        },
    }

    @classmethod
    def save_mlflow_state(
        cls, experiment_name: str, run_id: str = None, tracking_uri: str = None
    ) -> Any:
        """Save MLflow state for later retrieval."""
        cls.MLFLOW_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "experiment_name": experiment_name,
            "last_run_id": run_id,
            "tracking_uri": tracking_uri,
        }
        with open(cls.MLFLOW_STATE_FILE, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_mlflow_state(cls) -> Dict[str, Any]:
        """Load saved MLflow state."""
        if cls.MLFLOW_STATE_FILE.exists():
            try:
                with open(cls.MLFLOW_STATE_FILE, "r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                # State file corrupted or unreadable, use defaults
                pass
        return {"experiment_name": "jotty_ml", "last_run_id": None, "tracking_uri": None}

    def _show_leaderboard(self, cli: "JottyCLI") -> Any:
        """Display dataset leaderboard."""
        cli.renderer.header("Dataset Leaderboard")
        cli.renderer.info("")
        cli.renderer.info(
            "┌────────────────┬────────────────┬──────────┬──────────┬──────────┬─────────┐"
        )
        cli.renderer.info(
            "│    Dataset     │      Type      │ Baseline │   Best   │  Target  │ Samples │"
        )
        cli.renderer.info(
            "├────────────────┼────────────────┼──────────┼──────────┼──────────┼─────────┤"
        )
        for name, info in self.DATASET_LEADERBOARD.items():
            dtype = info["type"][:14]
            baseline = f"{info['baseline']*100:.1f}%"
            best = f"{info['best']*100:.1f}%"
            target = info["target"][:8]
            samples = info["samples"]
            cli.renderer.info(
                f"│ {name:<14} │ {dtype:<14} │ {baseline:>8} │ {best:>8} │ {target:<8} │ {samples:>7} │"
            )
        cli.renderer.info(
            "└────────────────┴────────────────┴──────────┴──────────┴──────────┴─────────┘"
        )
        cli.renderer.info("")
        cli.renderer.info("Baseline: Simple model (LogisticRegression/Ridge)")
        cli.renderer.info("Best: Known best score achievable with AutoML")

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute ML pipeline."""

        # Check for leaderboard flag
        if "leaderboard" in args.flags or "lb" in args.flags:
            self._show_leaderboard(cli)
            return CommandResult.ok(data=self.DATASET_LEADERBOARD)

        # Parse arguments
        dataset = args.positional[0] if args.positional else None
        target_col = args.flags.get("target", args.flags.get("t"))
        context = args.flags.get("context", args.flags.get("c", "ML prediction task"))
        iterations = int(args.flags.get("iterations", args.flags.get("i", "2")))

        # MLflow options
        use_mlflow = "mlflow" in args.flags or args.flags.get("mlflow") == "true"
        experiment_name = args.flags.get("experiment", args.flags.get("exp", "jotty_ml"))
        tracking_uri = args.flags.get("tracking-uri", args.flags.get("uri"))

        # Database options
        query = args.flags.get("query", args.flags.get("q"))
        connection = args.flags.get("connection", args.flags.get("conn"))
        db_type = args.flags.get("db-type", args.flags.get("db"))
        db_host = args.flags.get("host", "localhost")
        db_port = args.flags.get("port")
        db_name = args.flags.get("database", args.flags.get("dbname"))
        db_user = args.flags.get("user", args.flags.get("u"))
        db_password = args.flags.get("password", args.flags.get("p"))

        # Load from database if query provided
        if query:
            cli.renderer.info("Loading from database query...")
            try:
                X, y, target_name = await self._load_from_database(
                    query=query,
                    target_col=target_col,
                    connection=connection,
                    db_type=db_type,
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_password,
                    cli=cli,
                )
                if X is None:
                    return CommandResult.fail("Failed to load from database")
            except Exception as e:
                cli.renderer.error(f"Database query failed: {e}")
                return CommandResult.fail(str(e))
        elif not dataset:
            cli.renderer.error("Dataset required. Usage: /ml <dataset> [--target <column>]")
            cli.renderer.info("Examples:")
            cli.renderer.info("  /ml titanic                    # Seaborn titanic dataset")
            cli.renderer.info("  /ml iris                       # Seaborn iris dataset")
            cli.renderer.info("  /ml data.csv --target label    # Custom CSV")
            cli.renderer.info("  /ml --query 'SELECT * FROM users' --connection pg --target churn")
            cli.renderer.info("")
            cli.renderer.info("MLflow tracking:")
            cli.renderer.info("  /ml titanic --mlflow           # Enable MLflow")
            cli.renderer.info("  /ml titanic --mlflow --experiment myexp")
            cli.renderer.info("")
            cli.renderer.info("Other options:")
            cli.renderer.info("  /ml --leaderboard              # Show dataset leaderboard")
            return CommandResult.fail("Dataset required")
        else:
            # Load dataset from file/builtin
            cli.renderer.info(f"Loading dataset: {dataset}")
            try:
                X, y, target_name = await self._load_dataset(dataset, target_col, cli)
                if X is None:
                    return CommandResult.fail("Failed to load dataset")
            except Exception as e:
                cli.renderer.error(f"Failed to load dataset: {e}")
                return CommandResult.fail(str(e))

        cli.renderer.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
        cli.renderer.info(f"Target: {target_name}")

        # Run SwarmML pipeline
        cli.renderer.info("")
        cli.renderer.header("SwarmML Pipeline")
        cli.renderer.info(f"Context: {context}")
        cli.renderer.info(f"Feedback iterations: {iterations}")
        if use_mlflow:
            cli.renderer.info(f"MLflow: enabled (experiment: {experiment_name})")
        cli.renderer.info("")

        try:
            result = await self._run_swarm_ml(
                X,
                y,
                context,
                iterations,
                cli,
                use_mlflow=use_mlflow,
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                dataset_name=dataset or "database_query",
            )
            return CommandResult.ok(data=result)
        except Exception as e:
            cli.renderer.error(f"Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return CommandResult.fail(str(e))

    async def _load_from_database(
        self,
        query: str,
        target_col: str,
        connection: str = None,
        db_type: str = None,
        host: str = "localhost",
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        cli: "JottyCLI" = None,
    ) -> Any:
        """Load dataset from database using ConnectorX."""
        import pandas as pd

        # Resolve connection alias
        if connection and connection in self.DB_ALIASES:
            conn_config = self.DB_ALIASES[connection].copy()
            db_type = db_type or conn_config.get("db_type")
            host = conn_config.get("host", host)
            port = port or conn_config.get("port")

        # Get credentials from environment if not provided
        db_type = db_type or os.environ.get("JOTTY_DB_TYPE", "postgresql")
        database = database or os.environ.get("JOTTY_DB_NAME", "")
        user = user or os.environ.get("JOTTY_DB_USER", "")
        password = password or os.environ.get("JOTTY_DB_PASSWORD", "")

        if not database:
            cli.renderer.error("Database name required. Use --database or set JOTTY_DB_NAME")
            return None, None, None

        if not target_col:
            cli.renderer.error("Target column required for database queries. Use --target <column>")
            return None, None, None

        # Use ConnectorX loader
        try:
            from Jotty.core.capabilities.semantic.query.data_loader import (
                DataLoaderFactory,
                OutputFormat,
            )

            loader = DataLoaderFactory.create(
                db_type=db_type,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )

            cli.renderer.info(f"Executing query on {db_type}://{host}/{database}...")
            df = loader.load(query, output_format=OutputFormat.PANDAS)
            cli.renderer.info(f"Loaded {len(df)} rows from database")

        except ImportError:
            cli.renderer.error("ConnectorX not installed. Install with: pip install connectorx")
            return None, None, None
        except Exception as e:
            cli.renderer.error(f"Database error: {e}")
            return None, None, None

        # Prepare X, y
        if target_col not in df.columns:
            cli.renderer.error(f"Target column '{target_col}' not in query results")
            cli.renderer.info(f"Available columns: {list(df.columns)}")
            return None, None, None

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                X[col] = pd.Categorical(X[col]).codes

        # Fill missing values
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        return X, y, target_col

    async def _load_dataset(self, dataset: str, target_col: Optional[str], cli: "JottyCLI") -> Any:
        """Load dataset from various sources."""
        import pandas as pd

        dataset_lower = dataset.lower()

        # Check if it's a seaborn dataset
        if dataset_lower in self.SEABORN_DATASETS:
            import seaborn as sns

            df = sns.load_dataset(dataset_lower)
            cli.renderer.info(f"Loaded seaborn dataset: {dataset}")

            # Handle specific datasets
            if dataset_lower == "titanic":
                leaky_cols = ["alive", "adult_male", "who", "alone"]
                df = df.drop(columns=[c for c in leaky_cols if c in df.columns])
                df = df.dropna(subset=["age", "fare", "embarked"])
                target_col = target_col or "survived"

            elif dataset_lower == "iris":
                target_col = target_col or "species"

            elif dataset_lower == "tips":
                target_col = target_col or "tip"

            elif dataset_lower == "penguins":
                df = df.dropna()
                target_col = target_col or "species"

            elif dataset_lower == "diamonds":
                target_col = target_col or "price"
                if len(df) > 5000:
                    df = df.sample(5000, random_state=42)

            elif dataset_lower == "mpg":
                df = df.dropna()
                target_col = target_col or "mpg"

        # Check if it's a sklearn dataset
        elif dataset_lower in self.SKLEARN_DATASETS:
            from sklearn import datasets as sklearn_datasets

            if dataset_lower == "breast_cancer":
                data = sklearn_datasets.load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                target_col = target_col or "target"
                cli.renderer.info("Loaded sklearn breast_cancer (classification)")

            elif dataset_lower == "wine":
                data = sklearn_datasets.load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                target_col = target_col or "target"
                cli.renderer.info("Loaded sklearn wine (classification)")

            elif dataset_lower == "digits":
                data = sklearn_datasets.load_digits()
                df = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(64)])
                df["target"] = data.target
                target_col = target_col or "target"
                # Sample for speed
                if len(df) > 1000:
                    df = df.sample(1000, random_state=42)
                cli.renderer.info("Loaded sklearn digits (classification)")

            elif dataset_lower == "california":
                data = sklearn_datasets.fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                target_col = target_col or "target"
                # Sample for speed
                if len(df) > 5000:
                    df = df.sample(5000, random_state=42)
                cli.renderer.info("Loaded sklearn california_housing (regression)")

            elif dataset_lower == "diabetes":
                data = sklearn_datasets.load_diabetes()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                target_col = target_col or "target"
                cli.renderer.info("Loaded sklearn diabetes (regression)")

        elif Path(dataset).exists():
            # Load from file
            if dataset.endswith(".csv"):
                df = pd.read_csv(dataset)
            elif dataset.endswith(".parquet"):
                df = pd.read_parquet(dataset)
            elif dataset.endswith(".json"):
                df = pd.read_json(dataset)
            elif dataset.endswith(".xlsx") or dataset.endswith(".xls"):
                df = pd.read_excel(dataset)
            else:
                cli.renderer.error(f"Unsupported file format: {dataset}")
                return None, None, None

            cli.renderer.info(f"Loaded file: {dataset}")

            if not target_col:
                cli.renderer.error(
                    "Target column required for custom datasets. Use --target <column>"
                )
                return None, None, None

        else:
            cli.renderer.error(f"Dataset not found: {dataset}")
            cli.renderer.info("")
            cli.renderer.info("Available datasets:")
            cli.renderer.info(f"  Seaborn: {', '.join(self.SEABORN_DATASETS)}")
            cli.renderer.info(f"  Sklearn: {', '.join(self.SKLEARN_DATASETS)}")
            cli.renderer.info("  Files: .csv, .parquet, .json, .xlsx")
            cli.renderer.info("  Database: --query 'SQL' --connection pg --target col")
            return None, None, None

        # Prepare X, y
        if target_col not in df.columns:
            cli.renderer.error(f"Target column '{target_col}' not in dataset")
            cli.renderer.info(f"Available columns: {list(df.columns)}")
            return None, None, None

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                X[col] = pd.Categorical(X[col]).codes

        # Fill missing values
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        return X, y, target_col

    async def _run_swarm_ml(
        self,
        X: Any,
        y: Any,
        context: str,
        max_iterations: int,
        cli: "JottyCLI",
        use_mlflow: bool = False,
        experiment_name: str = "jotty_ml",
        tracking_uri: str = None,
        dataset_name: str = "unknown",
    ) -> Dict[str, Any]:
        """Run the full SwarmML pipeline with optional MLflow tracking."""
        import warnings

        import pandas as pd

        # Suppress all sklearn/lightgbm feature name warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*feature names.*")

        from sklearn.model_selection import KFold, StratifiedKFold

        # Import skills
        from Jotty.core.capabilities.skills.ml import (
            EnsembleSkill,
            FeatureEngineeringSkill,
            FeatureSelectionSkill,
            HyperoptSkill,
            LLMFeatureReasonerSkill,
            MLflowTrackerSkill,
            ModelSelectionSkill,
        )

        # Initialize MLflow tracker if enabled
        mlflow_tracker = None
        if use_mlflow:
            mlflow_tracker = MLflowTrackerSkill()
            await mlflow_tracker.init(tracking_uri=tracking_uri, experiment_name=experiment_name)
            run_name = f"{dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            await mlflow_tracker.start_run(run_name=run_name)
            cli.renderer.info(f"MLflow run started: {run_name}")

        # Detect problem type
        if y.nunique() <= 20 and y.nunique() / len(y) < 0.05:
            problem_type = "classification"
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            problem_type = "regression"
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        cli.renderer.info(f"Problem type: {problem_type}")

        # Log initial params to MLflow
        if mlflow_tracker:
            await mlflow_tracker.log_params(
                {
                    "dataset": dataset_name,
                    "problem_type": problem_type,
                    "n_samples": X.shape[0],
                    "n_features_original": X.shape[1],
                    "max_iterations": max_iterations,
                    "business_context": context[:200],
                }
            )

        # Initialize skills
        llm_skill = LLMFeatureReasonerSkill()
        await llm_skill.init()

        fe_skill = FeatureEngineeringSkill()
        await fe_skill.init()

        fs_skill = FeatureSelectionSkill()
        await fs_skill.init()

        ms_skill = ModelSelectionSkill()
        await ms_skill.init()

        ho_skill = HyperoptSkill()
        await ho_skill.init()

        ens_skill = EnsembleSkill()
        await ens_skill.init()

        best_score = 0
        best_iteration = 0
        best_model = None
        results = []
        feature_importance = {}

        for iteration in range(max_iterations):
            cli.renderer.info("")
            cli.renderer.subheader(f"Iteration {iteration + 1}/{max_iterations}")

            # Step 1: LLM Feature Reasoning
            if iteration == 0:
                cli.renderer.status("LLM Feature Reasoning (6 personas)...")
                llm_result = await llm_skill.execute(
                    X, y, problem_type=problem_type, business_context=context
                )
            else:
                cli.renderer.status("LLM Feedback Loop...")
                llm_result = await llm_skill.feedback_loop(
                    X_fe,  # Use engineered features for feedback
                    y,
                    feature_importance=feature_importance,
                    iteration=iteration,
                    problem_type=problem_type,
                    business_context=context,
                )

            X_llm = llm_result.data
            cli.renderer.info(f"  Applied: {llm_result.metrics.get('n_applied', 0)} LLM features")

            # Step 2: Feature Engineering
            cli.renderer.status("Feature Engineering...")
            fe_result = await fe_skill.execute(X_llm, y, problem_type=problem_type)
            X_fe = fe_result.data
            cli.renderer.info(f"  Features: {X_llm.shape[1]} → {X_fe.shape[1]}")

            # Step 3: Feature Selection (BOHB + PASHA + 12 methods)
            cli.renderer.status("Feature Selection (14 methods incl. BOHB/PASHA)...")
            fs_result = await fs_skill.execute(X_fe, y, problem_type=problem_type)
            X_fs = fs_result.data
            cli.renderer.info(f"  Selected: {X_fs.shape[1]} features")

            # Step 4: Model Selection
            cli.renderer.status("Model Selection (8+ algorithms)...")
            ms_result = await ms_skill.execute(X_fs, y, problem_type=problem_type)
            all_scores = ms_result.metadata.get("all_scores", {})
            cli.renderer.info(
                f"  Best model: {ms_result.metadata.get('best_model')} = {ms_result.metrics.get('score'):.4f}"
            )

            # Step 5: Hyperparameter Optimization
            cli.renderer.status("Hyperparameter Optimization...")
            ho_result = await ho_skill.execute(
                X_fs,
                y,
                problem_type=problem_type,
                all_scores=all_scores,
                model_ranking=ms_result.metadata.get("model_ranking", []),
                tune_all=True,
            )
            cli.renderer.info(f"  Best after tuning: {ho_result.metrics.get('score'):.4f}")

            # Step 6: Ensemble
            cli.renderer.status("Ensemble (multi-level stacking)...")
            ens_result = await ens_skill.execute(
                X_fs,
                y,
                problem_type=problem_type,
                optimized_model=ho_result.data,
                best_single_score=ho_result.metrics.get("score", 0),
                all_scores=ho_result.metadata.get("all_tuned_scores", all_scores),
            )

            iteration_score = ens_result.metrics.get("score", 0)
            cli.renderer.info(f"  Iteration {iteration + 1} Score: {iteration_score:.4f}")

            # Extract feature importance for next iteration
            feature_importance = EnsembleSkill.extract_feature_importance(
                ens_result.data, list(X_fs.columns)
            )

            # Log iteration metrics to MLflow
            if mlflow_tracker:
                await mlflow_tracker.log_metrics(
                    {
                        f"iter_{iteration+1}_score": iteration_score,
                        f"iter_{iteration+1}_n_features": X_fs.shape[1],
                        f"iter_{iteration+1}_llm_features": llm_result.metrics.get("n_applied", 0),
                    },
                    step=iteration,
                )

            results.append(
                {
                    "iteration": iteration + 1,
                    "score": iteration_score,
                    "features": X_fs.shape[1],
                    "strategy": ens_result.metrics.get("strategy", "unknown"),
                }
            )

            if iteration_score > best_score:
                best_score = iteration_score
                best_iteration = iteration + 1
                best_model = ens_result.data
                cli.renderer.success("  NEW BEST!")

        # Final summary with tables
        cli.renderer.info("")
        cli.renderer.header("Results Summary")

        # Iteration results table
        cli.renderer.info("")
        cli.renderer.info("┌─────────────┬──────────┬──────────┬─────────────────┐")
        cli.renderer.info("│  Iteration  │  Score   │ Features │    Strategy     │")
        cli.renderer.info("├─────────────┼──────────┼──────────┼─────────────────┤")
        for r in results:
            marker = " " if r["iteration"] == best_iteration else " "
            cli.renderer.info(
                f"│{marker} {r['iteration']:^9} │ {r['score']:^8.4f} │ {r['features']:^8} │ {r['strategy'][:15]:^15} │"
            )
        cli.renderer.info("└─────────────┴──────────┴──────────┴─────────────────┘")

        cli.renderer.info("")
        cli.renderer.info(f"Best Iteration: {best_iteration}")
        cli.renderer.info(f"Best Score: {best_score:.4f} ({best_score*100:.2f}%)")

        # Compare to leaderboard if dataset is known
        if dataset_name.lower() in self.DATASET_LEADERBOARD:
            lb_info = self.DATASET_LEADERBOARD[dataset_name.lower()]
            baseline = lb_info["baseline"]
            known_best = lb_info["best"]
            improvement = ((best_score - baseline) / baseline) * 100
            gap_to_best = ((known_best - best_score) / known_best) * 100

            cli.renderer.info("")
            cli.renderer.info("Leaderboard Comparison:")
            cli.renderer.info(f"  Baseline:     {baseline*100:.1f}%")
            cli.renderer.info(
                f"  Your Score:   {best_score*100:.1f}% ({'+' if improvement > 0 else ''}{improvement:.1f}% vs baseline)"
            )
            cli.renderer.info(f"  Known Best:   {known_best*100:.1f}% ({gap_to_best:.1f}% gap)")

        sorted_fi = []
        if feature_importance:
            sorted_fi = sorted(feature_importance.items(), key=lambda x: -x[1])[:10]
            cli.renderer.info("")
            cli.renderer.info("Top 10 Features:")
            cli.renderer.info("┌────────────────────────────────┬────────────┐")
            cli.renderer.info("│           Feature              │ Importance │")
            cli.renderer.info("├────────────────────────────────┼────────────┤")
            for feat, imp in sorted_fi:
                feat_display = feat[:30] if len(feat) > 30 else feat
                cli.renderer.info(f"│ {feat_display:<30} │ {imp:>10.4f} │")
            cli.renderer.info("└────────────────────────────────┴────────────┘")

        # Log final results to MLflow
        if mlflow_tracker:
            await mlflow_tracker.log_metrics(
                {
                    "best_score": best_score,
                    "best_iteration": best_iteration,
                    "final_n_features": X_fs.shape[1],
                }
            )

            # Log feature importance
            if feature_importance:
                await mlflow_tracker.log_feature_importance(feature_importance)

            # Log the best model
            if best_model is not None:
                model_uri = await mlflow_tracker.log_model(
                    best_model,
                    model_name="best_ensemble",
                    registered_name=f"{dataset_name}_model" if dataset_name != "unknown" else None,
                )
                if model_uri:
                    cli.renderer.info(f"Model logged to MLflow: {model_uri}")

            # End the run
            run_info = await mlflow_tracker.end_run()
            if run_info:
                cli.renderer.info(f"MLflow run completed: {run_info['run_id']}")
                cli.renderer.info(f"Artifact URI: {run_info['artifact_uri']}")

                # Save MLflow state for later retrieval
                MLCommand.save_mlflow_state(
                    experiment_name=experiment_name,
                    run_id=run_info["run_id"],
                    tracking_uri=tracking_uri,
                )

        # Prepare result
        result_data = {
            "best_score": best_score,
            "best_iteration": best_iteration,
            "problem_type": problem_type,
            "iterations": results,
            "top_features": sorted_fi[:10] if feature_importance else [],
        }

        # Add MLflow run ID if available
        if mlflow_tracker:
            try:
                result_data["mlflow_run_id"] = run_info.get("run_id") if run_info else None
            except (AttributeError, KeyError):
                # run_info invalid or missing
                result_data["mlflow_run_id"] = None

        return result_data

    def get_completions(self, partial: str) -> list:
        """Get dataset completions."""
        datasets = self.SEABORN_DATASETS + self.SKLEARN_DATASETS
        flags = [
            "--target",
            "--context",
            "--iterations",
            "--mlflow",
            "--leaderboard",
            "--experiment",
            "--tracking-uri",
            "--query",
            "--connection",
            "--db-type",
            "--host",
            "--port",
            "--database",
            "--user",
            "--password",
        ]
        all_completions = datasets + flags
        return [s for s in all_completions if s.startswith(partial)]
