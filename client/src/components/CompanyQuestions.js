import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './CompanyQuestions.css';


function CompanyQuestions(props) {
  const { currrollno } = props;
  const history = useNavigate();
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno, history]);
  
  const token = localStorage.getItem('token');
  const [company, setCompany] = useState('');
  const [questions, setQuestions] = useState([]);
  const [filter, setFilter] = useState('all');
  const [sort, setSort] = useState('latest');
  const [companyNames, setCompanyNames] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false); // Add loading state

  useEffect(() => {
    const fetchCompanyNames = async () => {
      try {
        const companyNamesResponse = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/experience`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        const companyNames = [...new Set(companyNamesResponse.data.map((experience) => experience.company_name))];
        setCompanyNames(companyNames);
        console.log("company names : ", companyNames);
      } catch (error) {
        console.error(error);
      }
    };

    fetchCompanyNames();
  }, []);

  useEffect(() => {
    fetchQuestions();
  }, [company, filter, sort]);

  const fetchQuestions = async () => {
    if (company) {
      setLoading(true); // Set loading to true when fetching questions
      try {
        let response;
        if (filter === 'all') {
          const technicalResponse = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/recent-technical-questions/${company}`, {
            headers: {
              Authorization: `Bearer ${token}`
            }
          });
          const hrResponse = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/recent-hr-questions/${company}`, {
            headers: {
              Authorization: `Bearer ${token}`
            }
          });
          let allQuestions = [...technicalResponse.data, ...hrResponse.data];
          if (sort === 'oldest') {
            allQuestions = allQuestions.reverse();
          }
          setQuestions(allQuestions);
        } else {
          response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/recent-${filter}-questions/${company}`, {
            headers: {
              Authorization: `Bearer ${token}`
            }
          });
          let filteredQuestions = response.data;
          if (sort === 'oldest') {
            filteredQuestions = filteredQuestions.reverse();
          }
          setQuestions(filteredQuestions);
        }
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false); // Set loading to false after fetching questions
      }
    }
  };

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
    if (searchTerm) {
      setLoading(true); 
    }
  };

  const handleCompanyChange = () => {
    setCompany(searchTerm);
    setSearchTerm("");
    setLoading(false)
  };

  const handleFilterChange = (event) => {
    setFilter(event.target.value);
  };

  const handleSortChange = (event) => {
    setSort(event.target.value);
  };

  const splitQuestions = (questionsString) => {
    return questionsString && typeof questionsString === 'string' ? questionsString.split('\n') : [];
  };

  return (
    <div className="container">
      <div className="left-sidebar">
        <h1 style={{ marginBottom: "20%", marginTop: "0px", fontSize: "4em", lineHeight: "0.8", fontWeight: "700", letterSpacing: "-0.02em" }}>Get Company Wise Questions</h1>
        <div style={{ display: "flex", marginBottom: "30%" }}>
          <input className='cqinput'
            type="text"
            placeholder="Enter a company name here..."
            value={searchTerm}
            onChange={handleSearchChange}
          />
          <button className="search-btn" onClick={handleCompanyChange}>
            Go-&gt;
          </button>
        </div>
      </div>
      <div className="right-content">
        <div className='filtercontainer'>
          <div className="filters">
            <button
              className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              All Questions
            </button>
            <button
              className={`filter-btn ${filter === 'technical' ? 'active' : ''}`}
              onClick={() => setFilter('technical')}
            >
              Technical
            </button>
            <button
              className={`filter-btn ${filter === 'hr' ? 'active' : ''}`}
              onClick={() => setFilter('hr')}
            >
              HR
            </button>
          </div>
          <div className="sort-filters">
            <button
              className={`sort-btn ${sort === 'latest' ? 'active' : ''}`}
              onClick={() => setSort('latest')}
            >
              Latest
            </button>
            <button
              className={`sort-btn ${sort === 'oldest' ? 'active' : ''}`}
              onClick={() => setSort('oldest')}
            >
              Oldest
            </button>
          </div>
        </div>
        {!searchTerm && !company ? (<div className="questions-list"> <div style={{display:"flex", justifyContent:"center", width: "100%"}}>
            <img src="/orange ghost.gif" alt="Loading..."  style={{maxWidth:"100%", alignSelf:"center"}}/>
            </div>
            </div>)
            :
            (
        <div className="questions-list">
         
            
          {loading ? (
            <div style={{display:"flex", justifyContent:"center", width: "100%"}}>
            <img src="https://icon-library.com/images/loading-icon-animated-gif/loading-icon-animated-gif-7.jpg" alt="Loading..."  style={{maxWidth:"250px", alignSelf:"center"}}/>
            </div>
          ) : (
            <React.Fragment>
              <h1 style={{ marginBottom: "20px", color: "#fac670", marginTop: "0px", fontSize: "4em", lineHeight: "0.8", fontWeight: "700", letterSpacing: "-0.02em" }}>Questions asked in {company}</h1>
              <ol>
                {questions.map((question, index) => (
                  <React.Fragment key={index}>
                    {question.technical_questions && typeof question.technical_questions === 'string' && splitQuestions(question.technical_questions).map((q, idx) => (
                      <li key={`${index}-${idx}`}>{q}</li>
                    ))}
                  </React.Fragment>
                ))}
                {questions.map((question, index) => (
                  <React.Fragment key={index}>
                    {question.hr_questions && typeof question.hr_questions === 'string' && splitQuestions(question.hr_questions).map((q, idx) => (
                      <li key={`${index}-${idx}`}>{q}</li>
                    ))}
                  </React.Fragment>
                ))}
              </ol>
            </React.Fragment>
          )}
        </div>
            )}
      </div>
    </div>
  );
}

export default CompanyQuestions;
