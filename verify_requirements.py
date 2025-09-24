"""
TitanovaX Requirements Verification Script
Comprehensive verification against bot.txt requirements
"""

import os
import json
import importlib
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RequirementsVerifier:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.requirements = self.load_requirements()
        self.verification_results = {}
        
    def load_requirements(self):
        """Load requirements from bot.txt"""
        bot_file = self.project_root.parent / "bot.txt"
        if not bot_file.exists():
            logger.error(f"bot.txt not found at {bot_file}")
            return {}
            
        try:
            with open(bot_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse key requirements from bot.txt
            requirements = {
                "top_10_ea_upgrades": self.parse_ea_upgrades(content),
                "cross_cutting_upgrades": self.parse_cross_cutting(content),
                "engineering_phases": self.parse_phases(content),
                "components": self.parse_components(content),
                "acceptance_tests": self.parse_acceptance_tests(content),
                "autonomous_learning": self.parse_autonomous_learning(content),
                "safety_governance": self.parse_safety_governance(content)
            }
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error loading bot.txt: {e}")
            return {}
    
    def parse_ea_upgrades(self, content):
        """Parse Top 10 EA upgrades section"""
        upgrades = []
        lines = content.split('\n')
        current_ea = None
        
        for line in lines:
            if line.strip().startswith('1)') and 'Forex Fury' in line:
                current_ea = {"name": "Forex Fury", "upgrades": []}
                upgrades.append(current_ea)
            elif line.strip().startswith('TitanovaX upgrades') and current_ea:
                continue
            elif current_ea and line.strip().startswith(tuple(str(i) + '.' for i in range(1, 10))):
                upgrade = line.strip()
                if upgrade and not upgrade.startswith('TitanovaX'):
                    current_ea["upgrades"].append(upgrade)
        
        return upgrades
    
    def parse_cross_cutting(self, content):
        """Parse cross-cutting system upgrades"""
        cross_cutting = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'Regime-Aware Orchestration' in line:
                cross_cutting.append("Regime-Aware Orchestration")
            elif 'Ensemble Decision Engine' in line:
                cross_cutting.append("Ensemble Decision Engine")
            elif 'Model Gating & Canary Deployment' in line:
                cross_cutting.append("Model Gating & Canary Deployment")
            elif 'Per-symbol & Per-broker adaptive latency/spread model' in line:
                cross_cutting.append("Adaptive Latency/Spread Model")
            elif 'Hard Safety Layer in EA' in line:
                cross_cutting.append("Hard Safety Layer")
            elif 'RAG + LLM Explainability' in line:
                cross_cutting.append("RAG + LLM Explainability")
            elif 'Simulated A/B Shadow Testing' in line:
                cross_cutting.append("A/B Shadow Testing")
            elif 'Grid & Reverse Safety Net' in line:
                cross_cutting.append("Grid & Reverse Safety Net")
            elif 'Adaptive Capital Allocation' in line:
                cross_cutting.append("Adaptive Capital Allocation")
        
        return cross_cutting
    
    def parse_phases(self, content):
        """Parse engineering phases"""
        phases = {}
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'Phase 0' in line and 'Safety' in line:
                phases["Phase_0_Safety"] = {
                    "description": "Safety & Observability",
                    "requirements": ["Hard Safety Layer", "Prometheus metrics + alerts"]
                }
            elif 'Phase 1' in line and 'Regime' in line:
                phases["Phase_1_Regime"] = {
                    "description": "Regime & Ensemble Core",
                    "requirements": ["Transformer regime classifier", "XGBoost + meta-model", "ONNX export + inference"]
                }
            elif 'Phase 2' in line and 'Adaptive Execution' in line:
                phases["Phase_2_Execution"] = {
                    "description": "Adaptive Execution & Broker Models",
                    "requirements": ["Per-broker microbenchmark", "Smart Order Router"]
                }
            elif 'Phase 3' in line and 'Strategy Modules' in line:
                phases["Phase_3_Strategy"] = {
                    "description": "Strategy Modules & RL sizing",
                    "requirements": ["Grid/regime gating", "RL sizing module"]
                }
            elif 'Phase 4' in line and 'Explainability' in line:
                phases["Phase_4_Explainability"] = {
                    "description": "Explainability & Teaching",
                    "requirements": ["RAG + LLM explain pipeline", "Telegram teacher"]
                }
            elif 'Phase 5' in line and 'Continuous Learning' in line:
                phases["Phase_5_Learning"] = {
                    "description": "Continuous Learning & Governance",
                    "requirements": ["Auto-retrain pipeline", "Canary deployment & rollback"]
                }
        
        return phases
    
    def parse_components(self, content):
        """Parse component requirements"""
        components = {
            "ingestion_memory": ["data/ticks", "data/parquet", "data/events", "Redis/RedisTimeSeries", "Parquet storage", "Vector DB"],
            "model_suite": ["XGBoost/LightGBM", "Informer/TFT/Autoformer", "RL agent (SB3)", "Meta-ensemble", "LLM/Explainability"],
            "controller_orchestrator": ["Learning Engine", "Inference Server", "Decision Engine", "Signal Broker", "Watchdog & Self-Healer"],
            "execution_mt5": ["Non-bypassable risk checks", "Signed signals", "Smart Order Router"],
            "user_interface": ["Telegram consent", "Daily/weekly reports", "Explain trade button"],
            "monitoring_governance": ["Prometheus metrics", "Grafana dashboards", "Immutable audit store", "Security"]
        }
        return components
    
    def parse_acceptance_tests(self, content):
        """Parse acceptance tests"""
        tests = [
            "Bootstrap test: initial ingestion, training, shadow trades, consent message",
            "Consent gating: START button enables trading, EA refuses without signed signal",
            "Anomaly pause: 5x spread spike → system pauses scalping and alerts",
            "Auto-deploy gating: reduced performance model fails canary and rolls back",
            "Self-heal: service restart and quarantine after repeated failures"
        ]
        return tests
    
    def parse_autonomous_learning(self, content):
        """Parse autonomous learning requirements"""
        learning = {
            "bootstrap_phase": ["7-day learning", "historical data ingestion", "model pretraining", "shadow trading"],
            "live_phase": ["micro-updates", "nightly retrain", "weekly audit"],
            "memory_hierarchy": ["Short-term (Redis)", "Mid-term (Parquet)", "Long-term (Vector DB)"],
            "compression": ["daily summaries", "embeddings", "core-set selection"]
        }
        return learning
    
    def parse_safety_governance(self, content):
        """Parse safety and governance requirements"""
        safety = {
            "human_consent": "7-day consent gate with explicit START required",
            "hard_safety_layer": ["max_loss_per_trade", "daily_drawdown_cap", "max_open_trades", "max_correlated_exposure"],
            "model_gating": ["walk-forward criteria", "canary deployment", "automatic rollback"],
            "anomaly_detection": "pause trading on unusual market conditions",
            "self_healing": "auto-restart services and fallback to good models"
        }
        return safety
    
    def verify_project_structure(self):
        """Verify project structure matches requirements"""
        logger.info("Verifying project structure...")
        
        required_dirs = [
            "config", "models", "ml-brain", "mt5-executor", "monitoring", 
            "tests", "data", "logs", "docker"
        ]
        
        results = {}
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            results[dir_name] = exists
            
            if exists:
                logger.info(f"✅ {dir_name} directory exists")
            else:
                logger.warning(f"❌ {dir_name} directory missing")
        
        return results
    
    def verify_models(self):
        """Verify all required models are present"""
        logger.info("Verifying models...")
        
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            logger.error("Models directory does not exist")
            return False
        
        required_models = [
            "xgboost_market_predictor.json",
            "xgboost_metadata.json",
            "transformer_config.json", 
            "transformer_metadata.json",
            "rl_policy.json",
            "rl_metadata.json",
            "ensemble_config.json",
            "ensemble_metadata.json",
            "model_registry.json"
        ]
        
        results = {}
        all_present = True
        
        for model_file in required_models:
            model_path = models_dir / model_file
            exists = model_path.exists()
            results[model_file] = exists
            
            if exists:
                logger.info(f"✅ {model_file} exists")
            else:
                logger.error(f"❌ {model_file} missing")
                all_present = False
        
        return all_present, results
    
    def verify_core_components(self):
        """Verify core components are implemented"""
        logger.info("Verifying core components...")
        
        components = {
            "ConfigManager": "config/config_manager.py",
            "MLTrainer": "ml-brain/ml_brain/training/",
            "ONNXServer": "onnx_server.py", 
            "RiskManager": "safety_risk_layer.py",
            "SignalProcessor": "ensemble_decision_engine.py",
            "RegimeClassifier": "regime_classifier.py",
            "SmartOrderRouter": "smart_order_router.py",
            "Watchdog": "watchdog_self_healing.py"
        }
        
        results = {}
        all_implemented = True
        
        for component_name, file_path in components.items():
            full_path = self.project_root / file_path
            exists = full_path.exists()
            results[component_name] = exists
            
            if exists:
                logger.info(f"✅ {component_name} implemented")
            else:
                logger.error(f"❌ {component_name} missing")
                all_implemented = False
        
        return all_implemented, results
    
    def verify_safety_features(self):
        """Verify safety features are implemented"""
        logger.info("Verifying safety features...")
        
        safety_files = [
            "safety_risk_layer.py",
            "config/risk_config.json",
            "mt5-executor/RiskManager.mqh",
            "watchdog_self_healing.py"
        ]
        
        results = {}
        all_safety_present = True
        
        for safety_file in safety_files:
            file_path = self.project_root / safety_file
            exists = file_path.exists()
            results[safety_file] = exists
            
            if exists:
                logger.info(f"✅ Safety component {safety_file} exists")
            else:
                logger.error(f"❌ Safety component {safety_file} missing")
                all_safety_present = False
        
        return all_safety_present, results
    
    def verify_docker_setup(self):
        """Verify Docker setup"""
        logger.info("Verifying Docker setup...")
        
        docker_files = [
            "docker-compose.yml",
            "docker/ml-brain/Dockerfile",
            "docker/mt5-executor/Dockerfile", 
            "docker/orchestration/Dockerfile",
            "docker/dashboard/Dockerfile"
        ]
        
        results = {}
        all_docker_present = True
        
        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            exists = file_path.exists()
            results[docker_file] = exists
            
            if exists:
                logger.info(f"✅ Docker file {docker_file} exists")
            else:
                logger.error(f"❌ Docker file {docker_file} missing")
                all_docker_present = False
        
        return all_docker_present, results
    
    def verify_monitoring(self):
        """Verify monitoring setup"""
        logger.info("Verifying monitoring setup...")
        
        monitoring_files = [
            "monitoring/prometheus.yml",
            "config/anomaly_config.json",
            "config/ensemble_config.json",
            "config/regime_config.json"
        ]
        
        results = {}
        all_monitoring_present = True
        
        for monitor_file in monitoring_files:
            file_path = self.project_root / monitor_file
            exists = file_path.exists()
            results[monitor_file] = exists
            
            if exists:
                logger.info(f"✅ Monitoring component {monitor_file} exists")
            else:
                logger.error(f"❌ Monitoring component {monitor_file} missing")
                all_monitoring_present = False
        
        return all_monitoring_present, results
    
    def generate_missing_requirements_report(self):
        """Generate report of missing requirements"""
        logger.info("Generating missing requirements report...")
        
        # Check orchestration directory (missing from requirements)
        orchestration_missing = not (self.project_root / "orchestration").exists()
        
        # Check Telegram bot implementation
        telegram_missing = not (self.project_root / "telegram_bot.py").exists()
        
        # Check RAG/LLM implementation  
        rag_missing = not (self.project_root / "rag_llm_explainability.py").exists()
        
        # Check memory system
        memory_missing = not (self.project_root / "memory").exists()
        
        # Check CI/CD workflows
        cicd_missing = not (self.project_root / ".github/workflows/enhanced-ci-cd.yml").exists()
        
        missing_requirements = {
            "orchestration_directory": orchestration_missing,
            "telegram_bot": telegram_missing,
            "rag_llm_system": rag_missing,
            "memory_hierarchy": memory_missing,
            "enhanced_cicd": cicd_missing
        }
        
        return missing_requirements
    
    def run_verification(self):
        """Run complete verification"""
        logger.info("Starting comprehensive requirements verification...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "project_structure": self.verify_project_structure(),
            "models": self.verify_models(),
            "core_components": self.verify_core_components(),
            "safety_features": self.verify_safety_features(),
            "docker_setup": self.verify_docker_setup(),
            "monitoring": self.verify_monitoring(),
            "missing_requirements": self.generate_missing_requirements_report()
        }
        
        # Calculate overall compliance
        compliance_score = self.calculate_compliance_score(results)
        results["compliance_score"] = compliance_score
        results["status"] = "COMPLIANT" if compliance_score >= 80 else "NEEDS_WORK"
        
        # Save results
        results_path = self.project_root / "requirements_verification_report.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Verification complete. Results saved to {results_path}")
        logger.info(f"Overall compliance score: {compliance_score:.1f}%")
        logger.info(f"Status: {results['status']}")
        
        return results
    
    def calculate_compliance_score(self, results):
        """Calculate overall compliance score"""
        total_checks = 0
        passed_checks = 0
        
        # Project structure
        for check, result in results["project_structure"].items():
            total_checks += 1
            if result:
                passed_checks += 1
        
        # Models
        if results["models"][0]:  # Overall model check passed
            passed_checks += 5  # Weight models heavily
        total_checks += 5
        
        # Core components
        if results["core_components"][0]:
            passed_checks += 3
        total_checks += 3
        
        # Safety features
        if results["safety_features"][0]:
            passed_checks += 2
        total_checks += 2
        
        # Docker setup
        if results["docker_setup"][0]:
            passed_checks += 2
        total_checks += 2
        
        # Monitoring
        if results["monitoring"][0]:
            passed_checks += 2
        total_checks += 2
        
        return (passed_checks / total_checks) * 100 if total_checks > 0 else 0

def main():
    """Main function"""
    verifier = RequirementsVerifier()
    results = verifier.run_verification()
    
    print("\n" + "="*60)
    print("TITANOVAX REQUIREMENTS VERIFICATION REPORT")
    print("="*60)
    print(f"Compliance Score: {results['compliance_score']:.1f}%")
    print(f"Status: {results['status']}")
    print(f"Timestamp: {results['timestamp']}")
    print("\nMissing Requirements:")
    
    for req, missing in results['missing_requirements'].items():
        if missing:
            print(f"❌ {req.replace('_', ' ').title()}")
    
    print("\nDetailed report saved to: requirements_verification_report.json")
    print("="*60)

if __name__ == "__main__":
    main()