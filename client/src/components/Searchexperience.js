import React, { useState, useEffect, useRef } from 'react';
import { Card, Form, Button , Modal} from 'react-bootstrap';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import Sentiment from 'sentiment';
import { Doughnut } from 'react-chartjs-2';
import 'chart.js/auto';
import 'chartjs-adapter-moment';
import './Searchexperience.css';

const sentiment = new Sentiment();

const calculateSentiment = (experiences) => {
  let totalScore = 0;
  let positiveCount = 0;
  let negativeCount = 0;
  let neutralCount = 0;

  experiences.forEach((experience) => {
    const result = sentiment.analyze(experience.overall_experience);
    totalScore += result.score;

    if (result.score > 0) {
      positiveCount++;
    } else if (result.score < 0) {
      negativeCount++;
    } else {
      neutralCount++;
    }
  });

  const totalCount = experiences.length;
  const positivePercentage = (positiveCount / totalCount) * 100;
  const negativePercentage = (negativeCount / totalCount) * 100;
  const neutralPercentage = (neutralCount / totalCount) * 100;

  return {
    sentimentScore: totalScore / totalCount,
    positivePercentage,
    negativePercentage,
    neutralPercentage,
  };
};

const getDifficultyLevel = (experiences) => {
  let totalScore = 0;

  experiences.forEach((experience) => {
    const result = sentiment.analyze(experience.overall_experience);
    totalScore += result.score;
  });

  if (totalScore / experiences.length < -2) return 'Difficult';
  if (totalScore / experiences.length > 2) return 'Easy';
  return 'Moderate';
};

const getHiringPercentage = (experiences) => {
  let hiredCount = 0;

  experiences.forEach((experience) => {
    if (experience.placed) {
      hiredCount++;
    }
  });

  const totalCount = experiences.length;
  return (hiredCount / totalCount) * 100;
};
const CompanyQuestionsModal = ({ show, handleClose, company }) => {
  const [technicalQuestions, setTechnicalQuestions] = useState([]);
  const [hrQuestions, setHRQuestions] = useState([]);

  useEffect(() => {
    const fetchRecentQuestions = async () => {
      try {
        const technicalResponse = await axios.get(`http://localhost:3001/experiences/recent-technical-questions/${company}`);
        setTechnicalQuestions(technicalResponse.data);

        const hrResponse = await axios.get(`http://localhost:3001/experiences/recent-hr-questions/${company}`);
        setHRQuestions(hrResponse.data);
      } catch (error) {
        console.error('Error fetching recent questions:', error);
      }
    };

    if (show) {
      fetchRecentQuestions();
    }
  }, [show, company]);

  return (
    <Modal show={show} onHide={handleClose}>
      <Modal.Header closeButton>
        <Modal.Title>Recent Questions for {company}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h5>Recent Technical Questions:</h5>
        <ul>
          {technicalQuestions.map((question, index) => (
            <li key={index}>{question}</li>
          ))}
        </ul>
        <h5>Recent HR Questions:</h5>
        <ul>
          {hrQuestions.map((question, index) => (
            <li key={index}>{question}</li>
          ))}
        </ul>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleClose}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

const QuestionsDropdown = ({ questions, onHide }) => {
  return (
    <div className="dropdown">
      <ul className="dropdown-content">
        {questions.map((question, index) => (
          <li key={index}>{question}</li>
        ))}
      </ul>
      <Button variant="secondary" onClick={onHide}>
        Close
      </Button>
    </div>
  );
};


const SearchExperience = (props) => {
  const {currrollno}=props;
  const history = useNavigate();
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno, history]);
  const inputRef = useRef(null);
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({});
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [experiences, setExperiences] = useState([]);
  const [loading, setLoading] = useState(true);
 
  const token = localStorage.getItem('token');
  const [search, setSearch]=useState(false)
  const [companyNames, setCompanyNames] = useState([]);
  const [years, setYears] = useState([]);
  const [roles, setRoles] = useState([]);
  const [showTechnicalQuestions, setShowTechnicalQuestions] = useState(false);
  const [showHRQuestions, setShowHRQuestions] = useState(false);
  const [selectedCompany, setSelectedCompany] = useState('');
  const [technicalQuestions, setTechnicalQuestions] = useState([]);
  const [hrQuestions, setHRQuestions] = useState([]);


  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`http://localhost:3001/experiences/experience`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setExperiences(response.data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)));
        setLoading(false);

        // Fetch unique company names, years, and roles from the database
        const companyNamesResponse = await axios.get(`http://localhost:3001/experiences/experience`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        //const coresponse = await axios.get('http://localhost:3001/experiences');
        const companyNames = [...new Set(companyNamesResponse.data.map((experience) => experience.company_name))];
        setCompanyNames(companyNames);
         console.log("company names : ", companyNames);
        const yearsResponse = await axios.get(`http://localhost:3001/experiences/years`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setYears(yearsResponse.data.map(year => year.year));

        const rolesResponse = await axios.get(`http://localhost:3001/experiences/roles`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setRoles(rolesResponse.data.map(role => role.role_offered));
      } catch (error) {
        console.error('Error fetching experiences:', error);
      }
    }

    fetchData();
  }, []); 

  const handleSearch = async (event) => {
    event.preventDefault();
    try {
     
      console.log("filters : ", filters);
      console.log("sortby : ", sortBy);
      console.log("sortOrder : ", sortOrder);
      const response = await axios.get(`http://localhost:3001/experiences/search`, {
        params: {
          query,
          filters,
          sortBy,
          sortOrder,
        },
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      query.length>0 ? setSearch(true) : setSearch(false);
    
      setExperiences(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error searching experiences:', error);
    }
  };

  const handleFilterChange = (event) => {
    const { name, value } = event.target;
    setFilters({...filters, [name]: value });
  };

  const handleSortChange = (event) => {
    setSortBy(event.target.value);
  };

  const handleSortOrderChange = (event) => {
    setSortOrder(event.target.value);
  };
  const renderHorizontalBar = (difficulty) => {
    const barLength = 10; // total length of the bar
    const filledChar = '█';
    const emptyChar = '▒';
    let filledLength = 0;
    let barColor = '';
  
    switch (difficulty) {
      case 'Easy':
        filledLength = 4;
        barColor = '#4CAF50';  // green
        break;
      case 'Moderate':
        filledLength = 7;
        barColor = '#FFEB3B';  // yellow
        break;
      case 'Difficult':
        filledLength = 10;
        barColor = '#F44336';  // red
        break;
      default:
        filledLength = 0;
        break;
    }
  
    const bar = filledChar.repeat(filledLength) + emptyChar.repeat(barLength - filledLength);
  
    return (
      <div className="horizontal-bar" >
        {difficulty}: {bar}
      </div>
    );
  };
  

  const CompanyhrQuestions = ({ companyName, questions, onHide, ishr }) => {
    console.log("in company question")
    console.log("questions : ", questions)
   // console.log(typeof(questions[0].hr_questions))
    // Function to split questions string into an array of individual questions
   // fetchRecenthrQuestions(companyName);
    const splitQuestions = (questionsString) => {
      console.log("called split function")
      return  questionsString && typeof questionsString==='string' ? questionsString.split('\n') : [] ;
    };
   console.log(ishr)
    return (
      <div class="hrdropdown">
  <div class="hrmenu-toggle" id="menu-toggle">       
   {ishr?
  <ul>
         {   questions.map((question, index) => (
                  
                  <React.Fragment key={index}>
                    { question.hr_questions && typeof question.hr_questions==='string' && splitQuestions(question.hr_questions).map((q, idx) => (
                      <li key={`${index}-${idx}`}>{q}</li>
                    ))}
                  </React.Fragment>
                ))
  }
          </ul>
          :
          {

          }
}
  </div>
  

</div>
    );
  };
  
  const CompanyQuestions = ({ companyName, questions, onHide, ishr }) => {
    console.log("in company question")
    console.log("questions : ", questions)
   // fetchRecenttechQuestions(companyName);
   // console.log(typeof(questions[0].hr_questions))
    // Function to split questions string into an array of individual questions
    const splitQuestions = (questionsString) => {
      console.log("called split function")
      return  questionsString && typeof questionsString==='string' ? questionsString.split('\n') : [] ;
    };
   console.log(ishr)
    return (
      <div class="quesdropdown">
  <div class="quesmenu-toggle" id="menu-toggle">
   
  <ul>
            
              { questions.map((question, index) => (
                  <React.Fragment key={index}>
                    {question.technical_questions && typeof question.technical_questions === 'string' && splitQuestions(question.technical_questions).map((q, idx) => (
                      <li key={`${index}-${idx}`}>{q}</li>
                    ))}
                  </React.Fragment>
                ))}
          </ul>
  </div>
  

</div>
    );
  };
  
  
  
  
  const RenderCompanyStats = () => {
   
    const companies = {};
    experiences.forEach(exp => {
      if (!companies[exp.company_name]) {
        companies[exp.company_name] = [];
      }
      companies[exp.company_name].push(exp);
    });
  
    const companyStats = {};
    for (let company in companies) {
      companyStats[company] = {
        sentiment: calculateSentiment(companies[company]),
        difficulty: getDifficultyLevel(companies[company]),
        hiring: getHiringPercentage(companies[company]),
       
          ctc: {
            low: Math.min(...companies[company].map(exp => exp.ctc)),
            avg: companies[company].reduce((sum, exp) => sum + exp.ctc, 0) / companies[company].length,
            high: Math.max(...companies[company].map(exp => exp.ctc))
          }
        
      };
    }
  
    const handleTechnicalQuestionsClick = (company) => {
      console.log("company tech: ", company)
      
      setSelectedCompany(company);
      // Fetch recent technical questions for the selected company from the backend
      // Update the technicalQuestions state
      const fetchRecenttechQuestions = async () => {
        try {
          const technicalResponse = await axios.get(`http://localhost:3001/experiences/recent-technical-questions/${company}`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
          
          setTechnicalQuestions(technicalResponse.data);
    
        } catch (error) {
          console.error('Error fetching recent questions:', error);
        }
      };
      fetchRecenttechQuestions();
     showTechnicalQuestions===false? setShowTechnicalQuestions(true) : setShowTechnicalQuestions(false);
    };
  
    const handleHRQuestionsClick = (company) => {
      console.log("in handle hr : ", company)
      const fetchRecenthrQuestions = async () => {
        try {
         
          const hrResponse = await axios.get(`http://localhost:3001/experiences/recent-hr-questions/${company}`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
          setHRQuestions(hrResponse.data);
        } catch (error) {
          console.error('Error fetching recent questions:', error);
        }
      };
      setSelectedCompany(company);
      // Fetch recent HR questions for the selected company from the backend
      // Update the hrQuestions state
      fetchRecenthrQuestions();
     showHRQuestions===false? setShowHRQuestions(true) : setShowHRQuestions(false);
      console.log("Show hr ", showHRQuestions)
    };
  
    return Object.keys(companyStats).map((company, index) => (
      <div key={index} style={{ marginTop: '20px' }} className='companystats-container'>
        <div className='company-title'>

        <h2 style={{paddingLeft:"20%"}}>{company}</h2>
        <div style={{  display:"flex", flexDirection:"row",float:"right"}}>
              <Button variant="primary" style={{marginBottom:"5px"}} onClick={() => handleTechnicalQuestionsClick(company)}>Recent Technical Questions</Button>{' '}
              <Button variant="primary" onClick={() => handleHRQuestionsClick(company)}>Recent HR Questions</Button>
            </div>
        </div>
        <div key={index} style={{ marginTop: '20px', display:"flex", flexDirection:"row", justifyContent:"space-between" }}>
          <div className="doughnut-chart-container">
          <Doughnut 
            data={{
              labels: ['Positive', 'Neutral', 'Negative'],
              datasets: [{
                data: [
                  companyStats[company].sentiment.positivePercentage,
                  companyStats[company].sentiment.neutralPercentage,
                  companyStats[company].sentiment.negativePercentage
                ],
                backgroundColor: ['green', 'grey', 'red']
              }]
            }}
            options={{
              maintainAspectRatio: false,
              responsive: true,
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: 'Sentiment Analysis',
                },
              },
              scales: {
                x: {
                  grid: {
                    display: false,
                  },
                },
                y: {
                  grid: {
                    display: false,
                  },
                },
              },
            }}
          />
        </div>
          <div style={{display:"flex", flexDirection:"column", justifyContent : "space-between", width : "40%"}}>
            <div style={{display:"flex", flexDirection:"row", justifyContent:"space-between"}}>Difficulty: {renderHorizontalBar(companyStats[company].difficulty)} </div>
            <div style={{paddingRight:"5px"}}>Hiring Percentage: {companyStats[company].hiring.toFixed(2)}%</div> 
            <div>CTC Range: {companyStats[company].ctc.low.toFixed(2)} - {companyStats[company].ctc.avg.toFixed(2)} - {companyStats[company].ctc.high.toFixed(2)}</div>
            
          </div>
        
        </div>
        {showTechnicalQuestions && selectedCompany === company && (
          <CompanyQuestions
            companyName={selectedCompany}
            questions={technicalQuestions}
            onHide={() => setShowTechnicalQuestions(false)}
            ishr={false}
          />
        )}
        {showHRQuestions && selectedCompany === company && (
          <CompanyhrQuestions
            companyName={selectedCompany}
            questions={hrQuestions}
            onHide={() => setShowHRQuestions(false)}
            ishr={true}
          />
        )}
      </div>
    ));
  };
  

  return (
    <div className='search-container'>
      <div className="search-filter-container">
        <Form.Group className="search-bar" style={{ position: 'relative', width: "50%", marginRight: "10px" }}>
          <Form.Control
            type="text"
            placeholder="Search by Company, Year, or Role"
            className="w-100"
            value={query}
            onChange={(event) => setQuery(event.target.value) }
            ref={inputRef}
            aria-label="Search experiences"
            aria-describedby="search-button"
            style={{ width: "95%" }}
          />
          <Button id="search-button" type="submit" onClick={handleSearch}  style={{ position: 'absolute', right: '0', top: '0', bottom: '0', height: "45px", padding: "5px", width: "100px" }}>
            Search
          </Button>
        </Form.Group>
        <Form.Group className="filter-group" style={{ width: "auto" }}>
          <Form.Control
            as="select"
            name="company_name"
            value={filters.company_name || ""}
            onChange={handleFilterChange}
          >
            <option value="">All Companies</option>
            {companyNames.map((name, index) => (
              <option key={index} value={name}>{name}</option>
            ))}
          </Form.Control>
        </Form.Group>
        <Form.Group>
          <Form.Control
            as="select"
            name="year"
            value={filters.year || ""}
            onChange={handleFilterChange}
          >
            <option value="">All Years</option>
            {years.map((year, index) => (
              <option key={index} value={year}>{year}</option>
            ))}
          </Form.Control>
        </Form.Group>
        <Form.Group>
          <Form.Control
            as="select"
            name="role_offered"
            value={filters.role_offered || ""}
            onChange={handleFilterChange}
          >
            <option value="">All Roles</option>
            {roles.map((role, index) => (
              <option key={index} value={role}>{role}</option>
            ))}
          </Form.Control>
        </Form.Group>
        <Form.Group>
          <Form.Control as="select" value={sortBy} onChange={handleSortChange}>
            <option value="created_at">Latest</option>
            <option value="company_name">Company Name</option>
            <option value="role_offered">Role</option>
          </Form.Control>
        </Form.Group>
        <Form.Group>
          <Form.Control as="select" value={sortOrder} onChange={handleSortOrderChange}>
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </Form.Control>
        </Form.Group>
      </div>
      <div className="experiences">
        {loading ? (
          <p>Loading...</p>
        ) : experiences.length === 0 ? (
          <p>No results found</p>
        ) : (
          <div className='experience-results'>
            {(search || filters.company_name) && RenderCompanyStats()}
            {experiences.map((experience) => (
              <ExperienceCard key={experience.experience_id} experience={experience} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
const ExperienceCard = ({ experience }) => {
  const { title, description, readMore } = experience;
  const company = experience.company_name;
  const rollNo = experience.rollno;
  const roleOffered = experience.role_offered;
  const overall_experience = experience.overall_experience;
  const truncatedOverallExperience = overall_experience.slice(0, 130) + '...';

  const sentimentResult = sentiment.analyze(experience.overall_experience);
  const sentimentColor = sentimentResult.score > 0 ? 'green' : sentimentResult.score < 0 ? 'red' : 'grey';

  const difficultyLevel = getDifficultyLevel([experience]);
  let difficultyGradient;

  switch (difficultyLevel) {
    case 'Easy':
      difficultyGradient = 'linear-gradient(to right, #b9e68c, #69db7c)';
      break;
    case 'Moderate':
      difficultyGradient = 'linear-gradient(to right, #ffeaa7, #ff9f43)';
      break;
    case 'Difficult':
      difficultyGradient = 'linear-gradient(to right, #ff7979, #eb4d4b)';
      break;
    default:
      difficultyGradient = 'linear-gradient(to right, #dcdde1, #b2bec3)';
      break;
  }

  return (
    <div className='card'>
    {/*}
    <Card className="experiencecard" style={{ width: "95%" }}>
      <Card.Body>
        <Card.Title style={{ fontSize: "x-large", fontWeight: "-moz-initial", paddingBottom:"2px" }}>{company}</Card.Title>
        <Card.Text style={{paddingTop:"3px"}}>Batch : {experience.batch}</Card.Text>
        <Card.Text style={{paddingTop:"3px"}}>Role Offered : {roleOffered}</Card.Text>
        <Card.Text style={{paddingTop:"3px"}}>{truncatedOverallExperience}</Card.Text>
        <div style={{ color: sentimentColor, paddingTop:"3px" }}>
          Sentiment: {sentimentResult.score > 0 ? 'Positive' : sentimentResult.score < 0 ? 'Negative' : 'Neutral'}
        </div>
        <div style={{ marginTop: '5px', marginBottom:'8px',display: 'flex', alignItems: 'center' }}>
          Difficulty : {difficultyLevel}
          <span 
            style={{ 
              backgroundImage: difficultyGradient,
              width: '50px', 
              height: '10px', 
              marginLeft: '10px', 
              display: 'inline-block',
              borderRadius: '5px'
            }}>
          </span>
        </div>
        <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
      </Card.Body>
    </Card>
          */}
<div class="face face1">
      <div class="content">
        <span class="stars"></span>
        <h2 class="java">{company}</h2>
        <p class="java">Educational Criteria : {experience.educational_criteria}</p>
        <p class="java">Overall Experience : {truncatedOverallExperience}</p>
       <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
      </div>
    </div>
    <div class="face face2">
      <h2>{company}</h2>
      <div className='face2-contents'>
      <h4>Role Offered : {roleOffered}</h4>
      <h4>Batch : {experience.batch}</h4>
      <div style={{ color: sentimentColor, paddingTop:"3px" }} className='sentiment'>
          Experience was : {sentimentResult.score > 0 ? 'Positive' : sentimentResult.score < 0 ? 'Negative' : 'Neutral'}
        </div>
        <div style={{ marginTop: '5px', marginBottom:'8px',display: 'flex', alignItems: 'center' }} className='difficulty'>
          Difficulty : {difficultyLevel}
          <span 
            style={{ 
              backgroundImage: difficultyGradient,
              width: '50px', 
              height: '10px', 
              marginLeft: '10px', 
              display: 'inline-block',
              borderRadius: '5px'
            }} className='bar'>
          </span>
        </div>
        
        </div>
    </div>
  </div>

  );
};

export default SearchExperience;
