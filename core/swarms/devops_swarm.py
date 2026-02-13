"""
DevOps Swarm - World-Class Infrastructure & Operations
=======================================================

Production-grade swarm for:
- Infrastructure as Code generation
- CI/CD pipeline design
- Container orchestration
- Cloud architecture
- Security hardening
- Monitoring setup

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                          DEVOPS SWARM                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  Infrastructure│  │   CI/CD        │  │   Container    │            │
│  │    Architect   │  │   Designer     │  │   Specialist   │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Security     │  │   Monitoring   │  │   Cost         │            │
│  │   Hardener     │  │   Specialist   │  │   Optimizer    │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     CONFIG GENERATOR                             │   │
│  │   Generates production-ready infrastructure configurations       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.devops_swarm import DevOpsSwarm, deploy

    # Full swarm
    swarm = DevOpsSwarm()
    result = await swarm.design_infrastructure(app_type="web", cloud="aws")

    # One-liner
    result = await deploy(app_name="myapp", cloud="gcp")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .base_swarm import (
    SwarmConfig, SwarmResult, AgentRole,
    register_swarm,
)
from .base import DomainSwarm, AgentTeam, _split_field
from .swarm_signatures import DevOpsSwarmSignature
from ..agents.base import DomainAgent, DomainAgentConfig, BaseSwarmAgent

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITALOCEAN = "digitalocean"
    KUBERNETES = "kubernetes"


class IaCTool(Enum):
    TERRAFORM = "terraform"
    PULUMI = "pulumi"
    CLOUDFORMATION = "cloudformation"
    CDK = "cdk"
    ANSIBLE = "ansible"


class CIProvider(Enum):
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    AZURE_PIPELINES = "azure_pipelines"


class ContainerPlatform(Enum):
    DOCKER = "docker"
    PODMAN = "podman"
    KUBERNETES = "kubernetes"
    ECS = "ecs"
    FARGATE = "fargate"


@dataclass
class DevOpsConfig(SwarmConfig):
    """Configuration for DevOpsSwarm."""
    cloud_provider: CloudProvider = CloudProvider.AWS
    iac_tool: IaCTool = IaCTool.TERRAFORM
    ci_provider: CIProvider = CIProvider.GITHUB_ACTIONS
    container_platform: ContainerPlatform = ContainerPlatform.KUBERNETES
    include_security: bool = True
    include_monitoring: bool = True
    include_cost_optimization: bool = True
    environment: str = "production"
    region: str = "us-east-1"

    def __post_init__(self):
        self.name = "DevOpsSwarm"
        self.domain = "devops"


@dataclass
class InfrastructureSpec:
    """Infrastructure specification."""
    compute: Dict[str, Any]
    networking: Dict[str, Any]
    storage: Dict[str, Any]
    database: Dict[str, Any]
    security_groups: List[Dict[str, Any]]
    load_balancers: List[Dict[str, Any]]


@dataclass
class PipelineSpec:
    """CI/CD pipeline specification."""
    stages: List[Dict[str, Any]]
    triggers: List[str]
    environments: List[str]
    secrets: List[str]
    yaml_config: str


@dataclass
class ContainerSpec:
    """Container configuration."""
    dockerfile: str
    compose_file: str
    kubernetes_manifests: Dict[str, str]
    registry: str
    image_name: str


@dataclass
class SecurityConfig:
    """Security configuration."""
    iam_policies: List[Dict[str, Any]]
    secrets_management: str
    network_policies: List[Dict[str, Any]]
    encryption: Dict[str, Any]
    compliance: List[str]


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    metrics: List[str]
    alerts: List[Dict[str, Any]]
    dashboards: List[Dict[str, Any]]
    logging: Dict[str, Any]
    tracing: Dict[str, Any]


@dataclass
class DevOpsResult(SwarmResult):
    """Result from DevOpsSwarm."""
    infrastructure: Optional[InfrastructureSpec] = None
    pipeline: Optional[PipelineSpec] = None
    container: Optional[ContainerSpec] = None
    security: Optional[SecurityConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    iac_code: Dict[str, str] = field(default_factory=dict)
    estimated_cost: str = ""
    deployment_steps: List[str] = field(default_factory=list)


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class InfrastructureDesignSignature(dspy.Signature):
    """Design cloud infrastructure.

    You are a CLOUD ARCHITECT. Design infrastructure that is:
    1. Highly available
    2. Scalable
    3. Cost-effective
    4. Secure
    5. Maintainable

    Follow cloud-native best practices.
    """
    app_type: str = dspy.InputField(desc="Type of application: web, api, microservices, etc.")
    cloud: str = dspy.InputField(desc="Cloud provider")
    requirements: str = dspy.InputField(desc="Application requirements")
    scale: str = dspy.InputField(desc="Expected scale: small, medium, large, enterprise")

    compute_spec: str = dspy.OutputField(desc="JSON compute specification")
    networking_spec: str = dspy.OutputField(desc="JSON networking specification")
    storage_spec: str = dspy.OutputField(desc="JSON storage specification")
    database_spec: str = dspy.OutputField(desc="JSON database specification")
    architecture_notes: str = dspy.OutputField(desc="Architecture decisions and notes")


class CICDDesignSignature(dspy.Signature):
    """Design CI/CD pipeline.

    You are a CI/CD EXPERT. Design pipelines that:
    1. Build reliably
    2. Test thoroughly
    3. Deploy safely
    4. Roll back quickly
    5. Scale efficiently

    Include all stages from commit to production.
    """
    app_type: str = dspy.InputField(desc="Type of application")
    ci_provider: str = dspy.InputField(desc="CI/CD platform")
    deployment_target: str = dspy.InputField(desc="Deployment target: kubernetes, ecs, etc.")
    environments: str = dspy.InputField(desc="Environments: dev, staging, prod")

    pipeline_stages: str = dspy.OutputField(desc="JSON list of pipeline stages")
    triggers: str = dspy.OutputField(desc="Pipeline triggers, separated by |")
    yaml_config: str = dspy.OutputField(desc="Complete CI/CD YAML configuration")
    best_practices: str = dspy.OutputField(desc="CI/CD best practices applied, separated by |")


class ContainerizationSignature(dspy.Signature):
    """Design containerization strategy.

    You are a CONTAINER SPECIALIST. Create containers that are:
    1. Lightweight
    2. Secure
    3. Reproducible
    4. Fast to build
    5. Easy to debug

    Follow Docker/Kubernetes best practices.
    """
    app_type: str = dspy.InputField(desc="Application type")
    language: str = dspy.InputField(desc="Programming language/runtime")
    platform: str = dspy.InputField(desc="Container platform")
    requirements: str = dspy.InputField(desc="Application requirements")

    dockerfile: str = dspy.OutputField(desc="Optimized Dockerfile")
    compose_file: str = dspy.OutputField(desc="Docker Compose file for local development")
    k8s_deployment: str = dspy.OutputField(desc="Kubernetes deployment manifest")
    k8s_service: str = dspy.OutputField(desc="Kubernetes service manifest")


class SecurityHardeningSignature(dspy.Signature):
    """Design security hardening.

    You are a SECURITY ENGINEER. Implement security that:
    1. Follows least privilege
    2. Encrypts data at rest and in transit
    3. Manages secrets securely
    4. Monitors for threats
    5. Meets compliance requirements

    Defense in depth approach.
    """
    infrastructure: str = dspy.InputField(desc="Infrastructure specification")
    cloud: str = dspy.InputField(desc="Cloud provider")
    compliance: str = dspy.InputField(desc="Compliance requirements: SOC2, HIPAA, PCI, etc.")

    iam_policies: str = dspy.OutputField(desc="JSON IAM policies")
    network_policies: str = dspy.OutputField(desc="JSON network security policies")
    encryption_config: str = dspy.OutputField(desc="Encryption configuration")
    secrets_management: str = dspy.OutputField(desc="Secrets management approach")
    security_recommendations: str = dspy.OutputField(desc="Security recommendations, separated by |")


class MonitoringSetupSignature(dspy.Signature):
    """Design monitoring and observability.

    You are an OBSERVABILITY EXPERT. Set up monitoring that:
    1. Tracks key metrics
    2. Alerts on anomalies
    3. Enables debugging
    4. Provides dashboards
    5. Supports SLOs/SLIs

    Full observability stack.
    """
    infrastructure: str = dspy.InputField(desc="Infrastructure specification")
    app_type: str = dspy.InputField(desc="Application type")
    sla_requirements: str = dspy.InputField(desc="SLA requirements")

    metrics: str = dspy.OutputField(desc="Key metrics to track, separated by |")
    alerts: str = dspy.OutputField(desc="JSON list of alert configurations")
    dashboard_config: str = dspy.OutputField(desc="Dashboard configuration")
    logging_config: str = dspy.OutputField(desc="Logging configuration")
    tracing_config: str = dspy.OutputField(desc="Distributed tracing configuration")


class IaCGenerationSignature(dspy.Signature):
    """Generate Infrastructure as Code.

    You are an IaC EXPERT. Generate code that is:
    1. Idempotent
    2. Modular
    3. Well-documented
    4. Follows conventions
    5. Version controlled

    Generate production-ready code.
    """
    infrastructure: str = dspy.InputField(desc="Infrastructure specification")
    iac_tool: str = dspy.InputField(desc="IaC tool: terraform, pulumi, etc.")
    cloud: str = dspy.InputField(desc="Cloud provider")

    main_config: str = dspy.OutputField(desc="Main configuration file")
    variables: str = dspy.OutputField(desc="Variables file")
    outputs: str = dspy.OutputField(desc="Outputs file")
    modules: str = dspy.OutputField(desc="Additional modules if needed")


# =============================================================================
# AGENTS
# =============================================================================

BaseDevOpsAgent = BaseSwarmAgent


class InfrastructureArchitect(BaseDevOpsAgent):
    """Designs cloud infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=InfrastructureDesignSignature)
        self._designer = dspy.ChainOfThought(InfrastructureDesignSignature)
        self.learned_context = learned_context

    async def design(
        self,
        app_type: str,
        cloud: str,
        requirements: str,
        scale: str = "medium"
    ) -> Dict[str, Any]:
        """Design infrastructure."""
        try:
            result = self._designer(
                app_type=app_type,
                cloud=cloud,
                requirements=requirements + (f"\n\nLearned Context:\n{self.learned_context}" if self.learned_context else ""),
                scale=scale
            )

            try:
                compute = json.loads(result.compute_spec)
            except Exception:
                compute = {}

            try:
                networking = json.loads(result.networking_spec)
            except Exception:
                networking = {}

            self._broadcast("infrastructure_designed", {
                'cloud': cloud,
                'scale': scale
            })

            return {
                'compute': compute,
                'networking': networking,
                'storage': result.storage_spec,
                'database': result.database_spec,
                'architecture_notes': str(result.architecture_notes)
            }

        except Exception as e:
            logger.error(f"Infrastructure design failed: {e}")
            return {'error': str(e)}


class CICDDesigner(BaseDevOpsAgent):
    """Designs CI/CD pipelines."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=CICDDesignSignature)
        self._designer = dspy.ChainOfThought(CICDDesignSignature)
        self.learned_context = learned_context

    async def design(
        self,
        app_type: str,
        ci_provider: str,
        deployment_target: str,
        environments: List[str]
    ) -> PipelineSpec:
        """Design CI/CD pipeline."""
        try:
            result = self._designer(
                app_type=app_type,
                ci_provider=ci_provider,
                deployment_target=deployment_target,
                environments=",".join(environments) + (f"\n\nLearned Context:\n{self.learned_context}" if self.learned_context else "")
            )

            try:
                stages = json.loads(result.pipeline_stages)
            except Exception:
                stages = []

            triggers = _split_field(result.triggers)

            self._broadcast("pipeline_designed", {
                'provider': ci_provider,
                'stages': len(stages)
            })

            return PipelineSpec(
                stages=stages,
                triggers=triggers,
                environments=environments,
                secrets=[],
                yaml_config=str(result.yaml_config)
            )

        except Exception as e:
            logger.error(f"CI/CD design failed: {e}")
            return PipelineSpec(
                stages=[],
                triggers=[],
                environments=environments,
                secrets=[],
                yaml_config=""
            )


class ContainerSpecialist(BaseDevOpsAgent):
    """Handles containerization."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=ContainerizationSignature)
        self._specialist = dspy.ChainOfThought(ContainerizationSignature)
        self.learned_context = learned_context

    async def containerize(
        self,
        app_type: str,
        language: str,
        platform: str,
        requirements: str = ""
    ) -> ContainerSpec:
        """Create container configuration."""
        try:
            result = self._specialist(
                app_type=app_type,
                language=language,
                platform=platform,
                requirements=(requirements or "Standard requirements") + (f"\n\nLearned Context:\n{self.learned_context}" if self.learned_context else "")
            )

            self._broadcast("containerized", {
                'platform': platform,
                'language': language
            })

            return ContainerSpec(
                dockerfile=str(result.dockerfile),
                compose_file=str(result.compose_file),
                kubernetes_manifests={
                    'deployment.yaml': str(result.k8s_deployment),
                    'service.yaml': str(result.k8s_service)
                },
                registry="",
                image_name=f"{app_type.lower()}-app"
            )

        except Exception as e:
            logger.error(f"Containerization failed: {e}")
            return ContainerSpec(
                dockerfile="",
                compose_file="",
                kubernetes_manifests={},
                registry="",
                image_name=""
            )


class SecurityHardener(BaseDevOpsAgent):
    """Handles security hardening."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=SecurityHardeningSignature)
        self._hardener = dspy.ChainOfThought(SecurityHardeningSignature)
        self.learned_context = learned_context

    async def harden(
        self,
        infrastructure: str,
        cloud: str,
        compliance: List[str]
    ) -> SecurityConfig:
        """Design security configuration."""
        try:
            result = self._hardener(
                infrastructure=infrastructure + (f"\n\nLearned Context:\n{self.learned_context}" if self.learned_context else ""),
                cloud=cloud,
                compliance=",".join(compliance) if compliance else "general"
            )

            try:
                iam_policies = json.loads(result.iam_policies)
            except Exception:
                iam_policies = []

            try:
                network_policies = json.loads(result.network_policies)
            except Exception:
                network_policies = []

            self._broadcast("security_configured", {
                'cloud': cloud,
                'compliance': compliance
            })

            return SecurityConfig(
                iam_policies=iam_policies,
                secrets_management=str(result.secrets_management),
                network_policies=network_policies,
                encryption={'config': str(result.encryption_config)},
                compliance=compliance
            )

        except Exception as e:
            logger.error(f"Security hardening failed: {e}")
            return SecurityConfig(
                iam_policies=[],
                secrets_management="",
                network_policies=[],
                encryption={},
                compliance=compliance
            )


class MonitoringSpecialist(BaseDevOpsAgent):
    """Sets up monitoring and observability."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=MonitoringSetupSignature)
        self._specialist = dspy.ChainOfThought(MonitoringSetupSignature)
        self.learned_context = learned_context

    async def setup(
        self,
        infrastructure: str,
        app_type: str,
        sla_requirements: str = ""
    ) -> MonitoringConfig:
        """Set up monitoring."""
        try:
            result = self._specialist(
                infrastructure=infrastructure + (f"\n\nLearned Context:\n{self.learned_context}" if self.learned_context else ""),
                app_type=app_type,
                sla_requirements=sla_requirements or "99.9% uptime"
            )

            metrics = _split_field(result.metrics)

            try:
                alerts = json.loads(result.alerts)
            except Exception:
                alerts = []

            self._broadcast("monitoring_configured", {
                'metrics': len(metrics),
                'alerts': len(alerts)
            })

            return MonitoringConfig(
                metrics=metrics,
                alerts=alerts,
                dashboards=[{'config': str(result.dashboard_config)}],
                logging={'config': str(result.logging_config)},
                tracing={'config': str(result.tracing_config)}
            )

        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return MonitoringConfig(
                metrics=[],
                alerts=[],
                dashboards=[],
                logging={},
                tracing={}
            )


class IaCGenerator(BaseDevOpsAgent):
    """Generates Infrastructure as Code."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=IaCGenerationSignature)
        self._generator = dspy.ChainOfThought(IaCGenerationSignature)
        self.learned_context = learned_context

    async def generate(
        self,
        infrastructure: str,
        iac_tool: str,
        cloud: str
    ) -> Dict[str, str]:
        """Generate IaC code."""
        try:
            result = self._generator(
                infrastructure=infrastructure + (f"\n\nLearned Context:\n{self.learned_context}" if self.learned_context else ""),
                iac_tool=iac_tool,
                cloud=cloud
            )

            self._broadcast("iac_generated", {
                'tool': iac_tool,
                'cloud': cloud
            })

            return {
                'main.tf': str(result.main_config),
                'variables.tf': str(result.variables),
                'outputs.tf': str(result.outputs),
                'modules': str(result.modules)
            }

        except Exception as e:
            logger.error(f"IaC generation failed: {e}")
            return {}


# =============================================================================
# DEVOPS SWARM
# =============================================================================

@register_swarm("devops")
class DevOpsSwarm(DomainSwarm):
    """
    World-Class DevOps Swarm.

    Provides comprehensive DevOps automation with:
    - Infrastructure design
    - CI/CD pipelines
    - Containerization
    - Security hardening
    - Monitoring setup
    - IaC generation
    """

    AGENT_TEAM = AgentTeam.define(
        (InfrastructureArchitect, "InfrastructureArchitect", "_infra_architect"),
        (CICDDesigner, "CICDDesigner", "_cicd_designer"),
        (ContainerSpecialist, "ContainerSpecialist", "_container_specialist"),
        (SecurityHardener, "SecurityHardener", "_security_hardener"),
        (MonitoringSpecialist, "MonitoringSpecialist", "_monitoring_specialist"),
        (IaCGenerator, "IaCGenerator", "_iac_generator"),
    )
    SWARM_SIGNATURE = DevOpsSwarmSignature

    def __init__(self, config: DevOpsConfig = None):
        super().__init__(config or DevOpsConfig())

    async def _execute_domain(
        self,
        app_name: str,
        **kwargs
    ) -> DevOpsResult:
        """Execute DevOps setup."""
        return await self.design_infrastructure(app_name=app_name, **kwargs)

    async def design_infrastructure(
        self,
        app_name: str = "myapp",
        app_type: str = "web",
        language: str = "python",
        requirements: str = "",
        scale: str = "medium",
        compliance: List[str] = None
    ) -> DevOpsResult:
        """
        Design complete infrastructure.

        Args:
            app_name: Application name
            app_type: Application type (web, api, microservices)
            language: Programming language
            requirements: Application requirements
            scale: Expected scale
            compliance: Compliance requirements

        Returns:
            DevOpsResult with complete configuration
        """
        config = self.config
        compliance = compliance or []

        logger.info(f"DevOpsSwarm starting: {app_name} on {config.cloud_provider.value}")

        return await self._safe_execute_domain(
            task_type='devops_setup',
            default_tools=['infra_design', 'cicd_design', 'container_setup'],
            result_class=DevOpsResult,
            execute_fn=lambda executor: self._execute_phases(
                executor, app_name, app_type, language, requirements, scale, compliance, config
            ),
            output_data_fn=lambda result: {
                'app_name': app_name,
                'cloud': config.cloud_provider.value,
                'has_infrastructure': result.infrastructure is not None,
                'has_pipeline': result.pipeline is not None and bool(getattr(result.pipeline, 'stages', [])),
                'has_container': result.container is not None and bool(getattr(result.container, 'dockerfile', '')),
                'has_security': result.security is not None,
                'has_monitoring': result.monitoring is not None,
                'iac_files_count': len(result.iac_code) if result.iac_code else 0,
                'deployment_steps_count': len(result.deployment_steps) if result.deployment_steps else 0,
                'execution_time': getattr(result, 'execution_time', 0),
            },
            input_data_fn=lambda: {
                'app_name': app_name,
                'app_type': app_type,
                'language': language,
                'requirements': requirements,
                'scale': scale,
                'compliance': compliance,
                'cloud_provider': config.cloud_provider.value,
                'iac_tool': config.iac_tool.value,
                'ci_provider': config.ci_provider.value,
                'container_platform': config.container_platform.value,
            },
        )

    async def _execute_phases(
        self,
        executor,
        app_name: str,
        app_type: str,
        language: str,
        requirements: str,
        scale: str,
        compliance: List[str],
        config: DevOpsConfig
    ) -> DevOpsResult:
        """Execute all DevOps phases using PhaseExecutor.

        Args:
            executor: PhaseExecutor instance for tracing and timing
            app_name: Application name
            app_type: Application type
            language: Programming language
            requirements: Application requirements
            scale: Expected scale
            compliance: Compliance requirements
            config: DevOps configuration

        Returns:
            DevOpsResult with complete configuration
        """
        # =================================================================
        # PHASE 1: INFRASTRUCTURE DESIGN
        # =================================================================
        infra_result = await executor.run_phase(
            1, "Infrastructure Design", "InfrastructureArchitect", AgentRole.PLANNER,
            self._infra_architect.design(
                app_type=app_type,
                cloud=config.cloud_provider.value,
                requirements=requirements or f"Standard {app_type} application",
                scale=scale
            ),
            input_data={'app_type': app_type, 'cloud': config.cloud_provider.value},
            tools_used=['infra_design'],
        )

        if isinstance(infra_result, dict) and 'error' in infra_result:
            return DevOpsResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=executor.elapsed(),
                error=infra_result['error']
            )

        infrastructure = InfrastructureSpec(
            compute=infra_result.get('compute', {}),
            networking=infra_result.get('networking', {}),
            storage={},
            database={},
            security_groups=[],
            load_balancers=[]
        )

        # =================================================================
        # PHASE 2: PARALLEL - CI/CD + Containers + Security
        # =================================================================
        parallel_tasks = [
            ("CICDDesigner", AgentRole.ACTOR, self._cicd_designer.design(
                app_type=app_type,
                ci_provider=config.ci_provider.value,
                deployment_target=config.container_platform.value,
                environments=["dev", "staging", "production"]
            ), ['cicd_design']),
            ("ContainerSpecialist", AgentRole.ACTOR, self._container_specialist.containerize(
                app_type=app_type,
                language=language,
                platform=config.container_platform.value,
                requirements=requirements
            ), ['container_setup']),
        ]

        if config.include_security:
            parallel_tasks.append(
                ("SecurityHardener", AgentRole.EXPERT, self._security_hardener.harden(
                    infrastructure=json.dumps(infra_result),
                    cloud=config.cloud_provider.value,
                    compliance=compliance
                ), ['security_harden'])
            )

        parallel_results = await executor.run_parallel(2, "CI/CD, Containers, Security", parallel_tasks)

        # Unpack parallel results with safe defaults for errors
        pipeline = parallel_results[0]
        if isinstance(pipeline, dict) and 'error' in pipeline:
            pipeline = PipelineSpec(stages=[], triggers=[], environments=[], secrets=[], yaml_config="")

        container = parallel_results[1]
        if isinstance(container, dict) and 'error' in container:
            container = ContainerSpec(dockerfile="", compose_file="", kubernetes_manifests={}, registry="", image_name="")

        security = None
        if config.include_security and len(parallel_results) > 2:
            security = parallel_results[2]
            if isinstance(security, dict) and 'error' in security:
                security = None

        # =================================================================
        # PHASE 3: MONITORING SETUP
        # =================================================================
        monitoring = None
        if config.include_monitoring:
            monitoring = await executor.run_phase(
                3, "Monitoring Setup", "MonitoringSpecialist", AgentRole.ACTOR,
                self._monitoring_specialist.setup(
                    infrastructure=json.dumps(infra_result),
                    app_type=app_type,
                    sla_requirements="99.9% uptime"
                ),
                input_data={'include_monitoring': config.include_monitoring},
                tools_used=['monitoring_setup'],
            )

        # =================================================================
        # PHASE 4: IAC GENERATION
        # =================================================================
        iac_code = await executor.run_phase(
            4, "Infrastructure as Code Generation", "IaCGenerator", AgentRole.ACTOR,
            self._iac_generator.generate(
                infrastructure=json.dumps(infra_result),
                iac_tool=config.iac_tool.value,
                cloud=config.cloud_provider.value
            ),
            input_data={'iac_tool': config.iac_tool.value},
            tools_used=['iac_generate'],
        )

        if isinstance(iac_code, dict) and 'error' in iac_code:
            iac_code = {}

        # =================================================================
        # BUILD RESULT
        # =================================================================
        deployment_steps = [
            f"1. Review and customize {config.iac_tool.value} configuration",
            f"2. Initialize {config.iac_tool.value}: `terraform init`",
            f"3. Plan infrastructure: `terraform plan`",
            f"4. Apply infrastructure: `terraform apply`",
            f"5. Build container: `docker build -t {app_name} .`",
            f"6. Push to registry",
            f"7. Deploy via {config.ci_provider.value} pipeline",
            "8. Verify monitoring dashboards",
            "9. Run smoke tests"
        ]

        logger.info(f"DevOpsSwarm complete: {app_name} infrastructure ready")

        return DevOpsResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                'iac_code': iac_code,
                'deployment_steps': deployment_steps,
                'app_name': app_name,
                'cloud': config.cloud_provider.value,
                'estimated_cost': "Varies based on scale and usage",
            },
            execution_time=executor.elapsed(),
            infrastructure=infrastructure,
            pipeline=pipeline,
            container=container,
            security=security,
            monitoring=monitoring,
            iac_code=iac_code,
            estimated_cost="Varies based on scale and usage",
            deployment_steps=deployment_steps
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def deploy(app_name: str, **kwargs) -> DevOpsResult:
    """
    One-liner DevOps setup.

    Usage:
        from core.swarms.devops_swarm import deploy
        result = await deploy("myapp", cloud="aws")
    """
    swarm = DevOpsSwarm()
    return await swarm.design_infrastructure(app_name=app_name, **kwargs)


def deploy_sync(app_name: str, **kwargs) -> DevOpsResult:
    """Synchronous DevOps setup."""
    return asyncio.run(deploy(app_name, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DevOpsSwarm',
    'DevOpsConfig',
    'DevOpsResult',
    'InfrastructureSpec',
    'PipelineSpec',
    'ContainerSpec',
    'SecurityConfig',
    'MonitoringConfig',
    'CloudProvider',
    'IaCTool',
    'CIProvider',
    'ContainerPlatform',
    'deploy',
    'deploy_sync',
    # Agents
    'InfrastructureArchitect',
    'CICDDesigner',
    'ContainerSpecialist',
    'SecurityHardener',
    'MonitoringSpecialist',
    'IaCGenerator',
]
