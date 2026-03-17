import argparse


class CLIParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Neuralvoid Terminal Assistant / Agent Deployer",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._build()

    # ─────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────
    @staticmethod
    def _max_iterations_type(value):
        ivalue = int(value)
        if ivalue < -1:
            raise argparse.ArgumentTypeError(
                "max-iterations must be -1 (infinite) or a non-negative integer"
            )
        return ivalue

    @staticmethod
    def _positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                "max-tokens must be a positive integer (> 0)"
            )
        return ivalue

    @staticmethod
    def _json_file_path(value):
        if not value.lower().endswith(".json"):
            raise argparse.ArgumentTypeError(
                f"File must have a .json extension: {value}"
            )
        return value

    # ─────────────────────────────────────────────
    def _build(self):
        self.parser.add_argument(
            "--deploy",
            type=str,
            metavar="PROMPT",
            default=None,
            help="Deploy headless agent with this prompt.\n"
            'Example: --deploy "Summarize the project and create a TODO list"',
        )

        self.parser.add_argument("--config", type=str, help="Path to config file")

        deploy_group = self.parser.add_argument_group(
            "headless agent options (only with --deploy)"
        )

        deploy_group.add_argument(
            "--status-file",
            type=self._json_file_path,  # <- validate JSON path extension only
            default="/tmp/neuralvoid/agent.status.json",
            metavar="PATH",
            help="Location of the agent status JSON file (must end with .json, default: %(default)s)",
        )

        deploy_group.add_argument(
            "--pid-file",
            type=str,
            default="/tmp/neuralvoid/agent.pid",
            metavar="PATH",
            help="Location of the agent PID file (default: %(default)s)",
        )

        deploy_group.add_argument(
            "--throttle-sec",
            type=float,
            default=1.5,
            metavar="SECONDS",
            help="Minimum time between status file updates (default: %(default)s s)",
        )

        deploy_group.add_argument(
            "--max-iterations",
            type=self._max_iterations_type,
            default=-1,
            metavar="N",
            help="Maximum number of iterations (-1 for infinite, default: %(default)s)",
        )

        deploy_group.add_argument(
            "--max-tokens",
            type=self._positive_int,
            default=12000,
            metavar="N",
            help="Maximum number of tokens per run (must be > 0, default: %(default)s)",
        )

    # ─────────────────────────────────────────────
    def parse(self):
        return self.parser.parse_args()
