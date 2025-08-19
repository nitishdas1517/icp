

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime

# Mock OpenAI client - replace with actual implementation
class MockOpenAI:
    """Mock OpenAI client for demonstration. Replace with actual OpenAI client."""
    
    async def chat_completions_create(self, model: str, messages: List[Dict], **kwargs):
        """Mock LLM response - replace with actual OpenAI API call"""
        # This is a simplified mock - in production, use actual OpenAI API
        system_msg = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
        user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
        
        # Mock intelligent responses based on system prompts
        if "research agent" in system_msg.lower():
            return await self._mock_research_response(user_msg)
        elif "company analysis" in system_msg.lower():
            return await self._mock_company_analysis(user_msg)
        elif "profile analysis" in system_msg.lower():
            return await self._mock_profile_analysis(user_msg)
        elif "icp matching" in system_msg.lower():
            return await self._mock_icp_analysis(user_msg)
        elif "decision synthesis" in system_msg.lower():
            return await self._mock_decision_synthesis(user_msg)
        elif "orchestrator" in system_msg.lower():
            return await self._mock_orchestrator_response(user_msg)
        
        return MockResponse("I need more context to provide an accurate response.")
    
    async def _mock_research_response(self, query: str):
        if "amit dugar" in query.lower():
            return MockResponse("""{
                "found": true,
                "current_company": "Boomerang (getboomerang.ai)",
                "current_title": "Chief Technology Officer",
                "previous_roles": [
                    {"company": "Mindtickle", "title": "Engineering Operations", "duration": "2020-2023"},
                    {"company": "BuyerAssist", "title": "CTO", "duration": "2023-2024"}
                ],
                "confidence": 0.95,
                "sources": ["linkedin_profile", "company_website", "crunchbase"]
            }""")
        return MockResponse('{"found": false, "confidence": 0.1}')
    
    async def _mock_company_analysis(self, query: str):
        if "buyerassist" in query.lower() and "boomerang" in query.lower():
            return MockResponse("""{
                "relationship_type": "rebranding",
                "is_job_change": false,
                "confidence": 0.92,
                "reasoning": "BuyerAssist rebranded to Boomerang in Q2 2024. Same leadership team, same office location, same core product. This is not a job change but a company rebrand.",
                "evidence": ["same_domain_ownership", "leadership_continuity", "product_continuity"]
            }""")
        elif "mindtickle" in query.lower():
            return MockResponse("""{
                "relationship_type": "different_company",
                "is_job_change": true,
                "confidence": 0.95,
                "reasoning": "Mindtickle and Boomerang are completely separate companies with different ownership, products, and leadership.",
                "evidence": ["different_founders", "different_products", "different_investors"]
            }""")
        return MockResponse('{"relationship_type": "unknown", "confidence": 0.3}')
    
    async def _mock_profile_analysis(self, query: str):
        return MockResponse("""{
            "career_progression": "upward",
            "role_evolution": "from operations to executive leadership",
            "seniority_level": "c_suite",
            "technical_focus": "engineering_leadership",
            "confidence": 0.88
        }""")
    
    async def _mock_icp_analysis(self, query: str):
        if "cto" in query.lower() or "chief technology officer" in query.lower():
            return MockResponse("""{
                "matches_icp": true,
                "confidence": 0.95,
                "reasoning": "CTO is a senior C-suite position in the engineering vertical, clearly matching the ICP criteria for senior engineering leadership.",
                "icp_score": 0.95
            }""")
        return MockResponse('{"matches_icp": false, "confidence": 0.7}')
    
    async def _mock_decision_synthesis(self, query: str):
        return MockResponse("""{
            "final_decision": {
                "isAJobChange": true,
                "isAnICP": true,
                "confidence": 0.91
            },
            "reasoning": "Based on analysis, person moved from Mindtickle to Boomerang (true job change) and holds CTO position (matches ICP)",
            "evidence_summary": ["career_progression_confirmed", "role_seniority_verified", "company_change_validated"]
        }""")
    
    async def _mock_orchestrator_response(self, query: str):
        return MockResponse("""{
            "next_actions": [
                {"agent": "research_agent", "task": "gather_current_profile_data"},
                {"agent": "company_analysis_agent", "task": "analyze_company_relationships"}
            ],
            "reasoning": "Need to gather current profile data first, then analyze company relationships to determine job change status."
        }""")

class MockResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]

class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)

class MockMessage:
    def __init__(self, content: str):
        self.content = content

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    COMPANY_ANALYSIS = "company_analysis"
    PROFILE_ANALYSIS = "profile_analysis"
    ICP_MATCHING = "icp_matching"
    DECISION_SYNTHESIS = "decision_synthesis"

@dataclass
class PersonProfile:
    fn: str
    ln: str
    company: str
    location: str
    email: str
    title: str
    icp: str

@dataclass
class AgentMessage:
    sender: AgentType
    recipient: AgentType
    content: Dict[str, Any]
    message_type: str
    timestamp: datetime

@dataclass
class ReasoningResult:
    fn: str
    ln: str
    title: str
    isAJobChange: bool
    isAnICP: bool
    currentCompany: str
    probableBusinessEmail: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""

class LLMAgent(ABC):
    """Base class for LLM-powered agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, system_prompt: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.message_bus = None
        self.llm_client = MockOpenAI()  # Replace with actual OpenAI client
        self.conversation_history = []
        
    def set_message_bus(self, message_bus):
        self.message_bus = message_bus
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method that uses LLM reasoning"""
        try:
            # Prepare the prompt with context
            user_prompt = self._prepare_prompt(data)
            
            # Get LLM response
            response = await self._get_llm_response(user_prompt)
            
            # Parse and validate response
            result = await self._process_llm_response(response, data)
            
            # Log the interaction
            self._log_interaction(user_prompt, response, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _get_llm_response(self, user_prompt: str) -> str:
        """Get response from LLM"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.llm_client.chat_completions_create(
            model="gpt-4",  # Use appropriate model
            messages=messages,
            temperature=0.1,  # Low temperature for consistent reasoning
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    async def call_agent(self, target_agent_type: AgentType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call another agent through the message bus"""
        if self.message_bus:
            return await self.message_bus.send_message(target_agent_type, data)
        return {}
    
    @abstractmethod
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        """Prepare the specific prompt for this agent"""
        pass
    
    @abstractmethod
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the LLM response and return structured data"""
        pass
    
    def _log_interaction(self, prompt: str, response: str, result: Dict[str, Any]):
        """Log the interaction for debugging and monitoring"""
        logger.info(f"Agent {self.agent_id} processed request with confidence: {result.get('confidence', 'N/A')}")

class OrchestratorAgent(LLMAgent):
    """Orchestrates the entire reasoning process"""
    
    def __init__(self):
        system_prompt = """
        You are an intelligent orchestrator agent responsible for coordinating multiple specialized agents 
        to determine job changes and ICP matching. You must:
        
        1. Analyze the input and determine what information is needed
        2. Decide which agents to call and in what order
        3. Synthesize results from multiple agents
        4. Handle edge cases and conflicting information
        
        Always provide your reasoning in JSON format with next_actions array and reasoning field.
        """
        super().__init__("orchestrator", AgentType.ORCHESTRATOR, system_prompt)
    
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        profile = data.get("profile")
        return f"""
        I need to analyze this profile for job changes and ICP matching:
        
        Name: {profile.get('fn')} {profile.get('ln')}
        Current Company: {profile.get('company')}
        Title: {profile.get('title')}
        Email: {profile.get('email')}
        Location: {profile.get('location')}
        ICP Criteria: {profile.get('icp')}
        
        Determine the optimal sequence of agent calls to get accurate results.
        What agents should I call and in what order?
        """
    
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            orchestration_plan = json.loads(response)
        except:
            # Fallback plan if JSON parsing fails
            orchestration_plan = {
                "next_actions": [
                    {"agent": "research_agent", "task": "gather_profile_data"},
                    {"agent": "company_analysis_agent", "task": "analyze_relationships"},
                    {"agent": "icp_matching_agent", "task": "check_icp_match"},
                    {"agent": "decision_synthesis_agent", "task": "synthesize_results"}
                ]
            }
        
        # Execute the orchestration plan
        results = {}
        profile = original_data.get("profile")
        
        # Step 1: Research current profile
        research_result = await self.call_agent(AgentType.RESEARCH, {
            "name": f"{profile['fn']} {profile['ln']}",
            "company": profile["company"],
            "task": "comprehensive_profile_research"
        })
        results["research"] = research_result
        
        # Step 2: Analyze company relationships
        company_analysis = await self.call_agent(AgentType.COMPANY_ANALYSIS, {
            "original_company": profile["company"],
            "current_company": research_result.get("current_company", profile["company"]),
            "person_name": f"{profile['fn']} {profile['ln']}"
        })
        results["company_analysis"] = company_analysis
        
        # Step 3: Analyze profile progression
        profile_analysis = await self.call_agent(AgentType.PROFILE_ANALYSIS, {
            "original_title": profile["title"],
            "current_title": research_result.get("current_title", profile["title"]),
            "career_data": research_result.get("previous_roles", [])
        })
        results["profile_analysis"] = profile_analysis
        
        # Step 4: Check ICP matching
        icp_analysis = await self.call_agent(AgentType.ICP_MATCHING, {
            "title": research_result.get("current_title", profile["title"]),
            "icp_criteria": profile["icp"],
            "career_level": profile_analysis.get("seniority_level", "unknown")
        })
        results["icp_analysis"] = icp_analysis
        
        # Step 5: Synthesize final decision
        final_decision = await self.call_agent(AgentType.DECISION_SYNTHESIS, {
            "profile": profile,
            "research": research_result,
            "company_analysis": company_analysis,
            "profile_analysis": profile_analysis,
            "icp_analysis": icp_analysis
        })
        
        return {
            "orchestration_plan": orchestration_plan,
            "agent_results": results,
            "final_result": final_decision,
            "confidence": final_decision.get("confidence", 0.5)
        }

class ResearchAgent(LLMAgent):
    """Researches current information about individuals and companies"""
    
    def __init__(self):
        system_prompt = """
        You are a research agent that gathers current information about individuals and companies.
        Use your knowledge to provide the most up-to-date information about:
        1. Current job positions and companies
        2. Recent career moves and changes
        3. Company relationships and structures
        
        Always return JSON with found, current_company, current_title, previous_roles, confidence, and sources fields.
        If you cannot find information, be honest about it and set found to false.
        """
        super().__init__("research", AgentType.RESEARCH, system_prompt)
    
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        name = data.get("name", "")
        company = data.get("company", "")
        
        return f"""
        I need current information about: {name}
        Last known company: {company}
        
        Please research and provide:
        1. Current company and position
        2. Recent job history (last 2-3 positions)
        3. Any notable career moves or changes
        4. Confidence level in the information
        
        Return as JSON format.
        """
    
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            research_data = json.loads(response)
            research_data["timestamp"] = datetime.now().isoformat()
            return research_data
        except json.JSONDecodeError:
            return {
                "found": False,
                "confidence": 0.1,
                "error": "Could not parse research results"
            }

class CompanyAnalysisAgent(LLMAgent):
    """Analyzes company relationships, mergers, acquisitions, and rebranding"""
    
    def __init__(self):
        system_prompt = """
        You are a company analysis agent that understands business relationships including:
        1. Company mergers and acquisitions
        2. Rebranding and name changes
        3. Parent-subsidiary relationships
        4. Spin-offs and corporate restructuring
        
        Determine if a person moving between companies represents a true job change or 
        if it's the same role due to corporate changes.
        
        Return JSON with relationship_type, is_job_change, confidence, reasoning, and evidence fields.
        """
        super().__init__("company_analysis", AgentType.COMPANY_ANALYSIS, system_prompt)
    
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        original_company = data.get("original_company", "")
        current_company = data.get("current_company", "")
        person_name = data.get("person_name", "")
        
        return f"""
        Analyze the relationship between these companies for {person_name}:
        
        Original Company: {original_company}
        Current Company: {current_company}
        
        Questions to answer:
        1. Is this a true job change or a company transformation (merger, acquisition, rebrand)?
        2. What type of relationship exists between these companies?
        3. What evidence supports this conclusion?
        4. How confident are you in this assessment?
        
        Consider factors like:
        - Company rebranding or name changes
        - Mergers and acquisitions
        - Subsidiary relationships
        - Leadership continuity
        - Product/service continuity
        """
    
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            return {
                "relationship_type": "unknown",
                "is_job_change": None,
                "confidence": 0.3,
                "reasoning": "Could not analyze company relationship"
            }

class ProfileAnalysisAgent(LLMAgent):
    """Analyzes career progression and role evolution"""
    
    def __init__(self):
        system_prompt = """
        You are a profile analysis agent that understands career progression patterns.
        Analyze how a person's role and seniority have evolved over time.
        
        Consider:
        1. Career trajectory (upward, lateral, downward)
        2. Role evolution and skill development
        3. Seniority level changes
        4. Industry transitions
        
        Return JSON with career_progression, role_evolution, seniority_level, technical_focus, and confidence.
        """
        super().__init__("profile_analysis", AgentType.PROFILE_ANALYSIS, system_prompt)
    
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        original_title = data.get("original_title", "")
        current_title = data.get("current_title", "")
        career_data = data.get("career_data", [])
        
        career_history = "\n".join([
            f"- {role.get('title', '')} at {role.get('company', '')} ({role.get('duration', '')})"
            for role in career_data
        ])
        
        return f"""
        Analyze this career progression:
        
        Original Title: {original_title}
        Current Title: {current_title}
        
        Career History:
        {career_history}
        
        Assess:
        1. Overall career trajectory
        2. Role evolution and responsibilities
        3. Current seniority level
        4. Technical vs management focus
        5. Leadership progression
        """
    
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "career_progression": "unknown",
                "seniority_level": "unknown",
                "confidence": 0.3
            }

class ICPMatchingAgent(LLMAgent):
    """Determines if a profile matches the Ideal Customer Profile"""
    
    def __init__(self):
        system_prompt = """
        You are an ICP matching agent that determines if a person fits specific 
        Ideal Customer Profile criteria.
        
        Analyze job titles, seniority levels, and responsibilities to determine matches.
        Consider context, industry variations, and role evolution.
        
        Return JSON with matches_icp, confidence, reasoning, and icp_score fields.
        """
        super().__init__("icp_matching", AgentType.ICP_MATCHING, system_prompt)
    
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        title = data.get("title", "")
        icp_criteria = data.get("icp_criteria", "")
        career_level = data.get("career_level", "")
        
        return f"""
        Determine if this profile matches the ICP criteria:
        
        Current Title: {title}
        Career Level: {career_level}
        ICP Criteria: {icp_criteria}
        
        Consider:
        1. Does the title match the seniority requirements?
        2. Is this person in the right vertical (engineering leadership)?
        3. What's the confidence level of this match?
        4. Are there any edge cases or special considerations?
        """
    
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "matches_icp": False,
                "confidence": 0.3,
                "reasoning": "Could not analyze ICP match"
            }

class DecisionSynthesisAgent(LLMAgent):
    """Synthesizes all agent results into final decisions"""
    
    def __init__(self):
        system_prompt = """
        You are a decision synthesis agent that combines results from multiple agents
        to make final determinations about job changes and ICP matching.
        
        Weight the confidence levels and reasoning from each agent.
        Handle conflicting information by considering the strength of evidence.
        
        Return JSON with final_decision, reasoning, confidence, and evidence_summary.
        """
        super().__init__("decision_synthesis", AgentType.DECISION_SYNTHESIS, system_prompt)
    
    def _prepare_prompt(self, data: Dict[str, Any]) -> str:
        profile = data.get("profile", {})
        research = data.get("research", {})
        company_analysis = data.get("company_analysis", {})
        profile_analysis = data.get("profile_analysis", {})
        icp_analysis = data.get("icp_analysis", {})
        
        return f"""
        Synthesize these agent results into final decisions:
        
        Original Profile:
        - Name: {profile.get('fn')} {profile.get('ln')}
        - Company: {profile.get('company')}
        - Title: {profile.get('title')}
        
        Research Results: {json.dumps(research, indent=2)}
        Company Analysis: {json.dumps(company_analysis, indent=2)}
        Profile Analysis: {json.dumps(profile_analysis, indent=2)}
        ICP Analysis: {json.dumps(icp_analysis, indent=2)}
        
        Make final decisions on:
        1. Is this a job change? (isAJobChange: boolean)
        2. Does this person match the ICP? (isAnICP: boolean)
        3. What's the current company and title?
        4. What's the probable business email?
        5. Overall confidence in these decisions
        
        Consider the confidence levels and evidence from each agent.
        """
    
    async def _process_llm_response(self, response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            synthesis = json.loads(response)
            
            # Extract key information for the final result
            profile = original_data.get("profile", {})
            research = original_data.get("research", {})
            
            result = ReasoningResult(
                fn=profile.get("fn", ""),
                ln=profile.get("ln", ""),
                title=research.get("current_title", profile.get("title", "")),
                isAJobChange=synthesis.get("final_decision", {}).get("isAJobChange", False),
                isAnICP=synthesis.get("final_decision", {}).get("isAnICP", False),
                currentCompany=research.get("current_company", profile.get("company", "")),
                probableBusinessEmail=self._generate_business_email(
                    profile.get("fn", ""),
                    profile.get("ln", ""),
                    research.get("current_company", "")
                ),
                confidence=synthesis.get("confidence", 0.5),
                reasoning=synthesis.get("reasoning", "")
            )
            
            return asdict(result)
            
        except json.JSONDecodeError:
            # Fallback result
            profile = original_data.get("profile", {})
            return asdict(ReasoningResult(
                fn=profile.get("fn", ""),
                ln=profile.get("ln", ""),
                title=profile.get("title", ""),
                isAJobChange=False,
                isAnICP=False,
                currentCompany=profile.get("company", ""),
                confidence=0.3,
                reasoning="Could not synthesize agent results"
            ))
    
    def _generate_business_email(self, first_name: str, last_name: str, company: str) -> str:
        """Generate probable business email based on company domain"""
        if not all([first_name, last_name, company]):
            return None
        
        # Extract domain from company name
        if "." in company:
            domain = company
        else:
            domain = f"{company.lower().replace(' ', '')}.com"
        
        # Generate email pattern
        first = first_name.lower()
        last = last_name.lower()
        
        # Most common pattern: first@domain or first.last@domain
        if len(first) <= 4:
            return f"{first}@{domain}"
        else:
            return f"{first}.{last}@{domain}"

class MessageBus:
    """Handles communication between agents"""
    
    def __init__(self):
        self.agents = {}
        self.message_log = []
    
    def register_agent(self, agent: LLMAgent):
        self.agents[agent.agent_type] = agent
        agent.set_message_bus(self)
        logger.info(f"Registered agent: {agent.agent_type}")
    
    async def send_message(self, target_type: AgentType, data: Dict[str, Any]) -> Dict[str, Any]:
        if target_type in self.agents:
            # Log the message
            message = AgentMessage(
                sender=AgentType.ORCHESTRATOR,  # Default sender
                recipient=target_type,
                content=data,
                message_type="request",
                timestamp=datetime.now()
            )
            self.message_log.append(message)
            
            # Send to target agent
            result = await self.agents[target_type].process(data)
            
            # Log the response
            response_message = AgentMessage(
                sender=target_type,
                recipient=AgentType.ORCHESTRATOR,
                content=result,
                message_type="response",
                timestamp=datetime.now()
            )
            self.message_log.append(response_message)
            
            return result
        
        logger.error(f"Agent {target_type} not found")
        return {"error": f"Agent {target_type} not found"}

class LLMMultiAgentReasoningSystem:
    """Main system that orchestrates LLM-powered agents"""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize and register all LLM agents"""
        agents = [
            OrchestratorAgent(),
            ResearchAgent(),
            CompanyAnalysisAgent(),
            ProfileAnalysisAgent(),
            ICPMatchingAgent(),
            DecisionSynthesisAgent()
        ]
        
        for agent in agents:
            self.message_bus.register_agent(agent)
        
        logger.info(f"Initialized {len(agents)} LLM agents")
    
    async def process_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a profile using LLM-powered multi-agent reasoning"""
        logger.info(f"Processing profile: {profile_data.get('fn')} {profile_data.get('ln')}")
        
        try:
            # Start with the orchestrator
            result = await self.message_bus.send_message(AgentType.ORCHESTRATOR, {
                "profile": profile_data
            })
            
            # Extract the final result
            final_result = result.get("final_result", {})
            
            logger.info(f"Processing complete with confidence: {final_result.get('confidence', 0)}")
            return final_result
            
        except Exception as e:
            logger.error(f"System error: {str(e)}")
            return {
                "error": str(e),
                "fn": profile_data.get("fn", ""),
                "ln": profile_data.get("ln", ""),
                "confidence": 0.0
            }
    
    def get_message_log(self) -> List[AgentMessage]:
        """Get the complete message log for debugging"""
        return self.message_bus.message_log

# Example usage and testing
async def main():
    """Test the LLM-based multi-agent system"""
    system = LLMMultiAgentReasoningSystem()
    
    print("=== LLM-Based Multi-Agent Reasoning System Test ===\n")
    
    # Test Case 1: True Job Change
    print("=== Test Case 1: True Job Change ===")
    test_case_1 = {
        "fn": "Amit",
        "ln": "Dugar",
        "company": "Mindtickle",
        "location": "Pune",
        "email": "amit.dugar@mindtickle.com",
        "title": "Engineering Operations",
        "icp": "The person has to be in senior position in Engineer Vertical like VP Engineering, CTO, Research Fellow"
    }
    
    result_1 = await system.process_profile(test_case_1)
    print(json.dumps(result_1, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Test Case 2: No Job Change (Rebranding)
    print("=== Test Case 2: No Job Change (Rebranding) ===")
    test_case_2 = {
        "fn": "Amit",
        "ln": "Dugar",
        "company": "BuyerAssist",
        "location": "Pune",
        "email": "amit.dugar@buyerassist.io",
        "title": "CTO",
        "icp": "The person has to be in senior position in Engineer Vertical like VP Engineering, CTO, Research Fellow"
    }
    
    result_2 = await system.process_profile(test_case_2)
    print(json.dumps(result_2, indent=2))
    
    print("\n=== Message Flow Analysis ===")
    message_log = system.get_message_log()
    for i, msg in enumerate(message_log[-6:], 1):  # Show last 6 messages
        print(f"{i}. {msg.sender.value} → {msg.recipient.value}: {msg.message_type}")

if __name__ == "__main__":
    asyncio.run(main())