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
            nargs="?",
            const="",
            default=None,
            metavar="[PROMPT]",
            help="Deploy headless agent.\n"
            '  - With prompt: --deploy "Summarize the project and create a TODO list"\n'
            "  - Without prompt: --deploy (starts in task-ready / autonomous state)",
        )

        self.parser.add_argument("--config", type=str, help="Path to config file")

        deploy_group = self.parser.add_argument_group(
            "headless agent options (only with --deploy)"
        )

        deploy_group.add_argument(
            "--agent",
            type=str,
            default=None,
            metavar="AGENT_ID",
            help="Specify which agent to deploy (default: the default agent in config).",
        )

        # [NEW] Multi-agent deployment support.
        # Accepts a comma-separated list of agent IDs.  When provided the Hub
        # is started automatically and agents communicate via WebSocket.
        deploy_group.add_argument(
            "--agents",
            type=str,
            default=None,
            metavar="ID1,ID2,...",
            help=(
                "Deploy multiple agents (comma-separated IDs).\n"
                "A central AgentHub is started automatically so agents\n"
                "can communicate with each other over WebSocket.\n"
                '  Example: --agents "agent_alpha,agent_beta"'
            ),
        )

        deploy_group.add_argument(
            "--hub-port",
            type=int,
            default=8770,
            metavar="PORT",
            help="Port for the AgentHub WebSocket server (default: %(default)s)",
        )

        deploy_group.add_argument(
            "--status-file",
            type=self._json_file_path,
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
