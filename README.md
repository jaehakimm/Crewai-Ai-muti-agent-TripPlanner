# CrewAI Project with Serper Integration 🤖

# Agent ai for Planing Trip 5 day Bangkok thailand 🤖

## Overview

This project demonstrates the implementation of CrewAI with Serper API integration for advanced internet data search capabilities. The system uses OpenAI's LLM as agents and incorporates user feedback for improved results.

<p align="center">
  <img src="crewAI-mindmap.png" alt="System Architecture" width="800">
</p>

## Features

- 🔍 Intelligent web searching using Serper API
- 🤖 Multiple AI agents powered by OpenAI
- 💡 Interactive feedback system for result optimization
- 🔄 Dynamic task allocation and management
- 📊 Customizable search parameters

## Installation

1. Install required packages:

```bash
pip install crewai crewai-tools
```

2. Set up your environment variables:

```bash
export SERPER_API_KEY="your-serper-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperTool

# Initialize your search tool
search_tool = SerperTool()

# Create an agent
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert at internet research",
    tools=[search_tool]
)

# Create tasks
research_task = Task(
    "Research and analyze [destination of choice] as a travel destination. "
        "Include information about:\n"
        "- Best time to visit and weather patterns\n"
        "- Must-see attractions and hidden gems\n"
        "- Local cuisine and dining recommendations\n"
        "- Transportation options and getting around\n"
        "- Accommodation options for different budgets\n"
        "- Cultural customs and etiquette\n"
        "- Safety considerations and practical tips\n"
        "Make sure to check with a human if the research is comprehensive enough before finalizing."
    agent=researcher
)

# Create and run your crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)

result = crew.kickoff()
```

## Screenshots when user can feedback for good result as you want

<p align="center">
  <img src="Screenshot 2024-11-09 172903.png" alt="อินเตอร์เฟซการตอบกลับ" width="800">
  <img src="Screenshot 2024-11-09 172932.png" alt="อินเตอร์เฟซการตอบกลับ" width="800">
  <img src="Screenshot 2024-11-09 173224.png" alt="อินเตอร์เฟซการตอบกลับ" width="800">
</p>



---

<a name="thai-version"></a>

# โครงการ CrewAI พร้อมการผสานรวม Serper 🤖
# Ai วางแผนเที่ยวในกรุงเทพ ประเทศไทย 5 วัน 🤖

<p align="center">
  <img src="crewAI-mindmap.png" alt="ภาพตัวอย่างโครงการ" width="600">
</p>

## ภาพรวม

โครงการนี้แสดงให้เห็นถึงการใช้งาน CrewAI ร่วมกับ Serper API สำหรับความสามารถในการค้นหาข้อมูลอินเทอร์เน็ตขั้นสูง ระบบใช้ OpenAI LLM เป็นตัวแทน AI และรวมการตอบกลับจากผู้ใช้เพื่อปรับปรุงผลลัพธ์

## คุณสมบัติ

- 🔍 การค้นหาเว็บอัจฉริยะด้วย Serper API
- 🤖 ตัวแทน AI หลายตัวที่ขับเคลื่อนด้วย OpenAI
- 💡 ระบบการตอบกลับแบบโต้ตอบเพื่อเพิ่มประสิทธิภาพผลลัพธ์
- 🔄 การจัดสรรและจัดการงานแบบไดนามิก
- 📊 พารามิเตอร์การค้นหาที่ปรับแต่งได้

## การติดตั้ง

1. ติดตั้งแพ็คเกจที่จำเป็น:

```bash
pip install crewai crewai-tools
```

2. ตั้งค่าตัวแปรสภาพแวดล้อม:

```bash
export SERPER_API_KEY="your-serper-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## ตัวอย่างการใช้งาน

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperTool

# เริ่มต้นเครื่องมือค้นหา
search_tool = SerperTool()

# สร้างตัวแทน
researcher = Agent(
    role='Travel Research Specialist',
    goal='Discover and analyze unique travel destinations, local experiences, and practical travel information',
    backstory=(
        "You are an experienced Travel Research Specialist with years of expertise in destination analysis. "
        "You have traveled to over 50 countries and specialize in uncovering hidden gems, "
        "understanding local cultures, and identifying authentic experiences. "
        "Your strength lies in combining practical travel logistics with cultural insights "
        "to create comprehensive travel guides."
    ),
    tools=[search_tool]
)

# สร้างงาน
research_task = Task(
        "Research and analyze [destination of choice] as a travel destination. "
        "Include information about:\n"
        "- Best time to visit and weather patterns\n"
        "- Must-see attractions and hidden gems\n"
        "- Local cuisine and dining recommendations\n"
        "- Transportation options and getting around\n"
        "- Accommodation options for different budgets\n"
        "- Cultural customs and etiquette\n"
        "- Safety considerations and practical tips\n"
        "Make sure to check with a human if the research is comprehensive enough before finalizing."
    agent=researcher
)

# สร้างและรันทีมงาน
crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)

result = crew.kickoff()
```

## ภาพหน้าจอตอนที่เราต้อง Feedback agent เกี่ยวกับผลลัพธ์

<p align="center">
  <img src="Screenshot 2024-11-09 172903.png" alt="อินเตอร์เฟซการตอบกลับ" width="800">
  <img src="Screenshot 2024-11-09 172932.png" alt="อินเตอร์เฟซการตอบกลับ" width="800">
  <img src="Screenshot 2024-11-09 173224.png" alt="อินเตอร์เฟซการตอบกลับ" width="800">
</p>