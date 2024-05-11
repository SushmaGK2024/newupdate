import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css';
import { useNavigate } from 'react-router-dom';
const linkStyle = {
  textDecoration: 'none',
  color: 'black',
  fontSize: '16px',
  marginBottom: '3px',
};

const Sidebar = (props) => {
  
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);

  const toggleProfileDropdown = () => {
    setProfileDropdownOpen(!profileDropdownOpen);
  };

  const [dropdownZIndex, setDropdownZIndex] = useState(1);

  useEffect(() => {
    if (profileDropdownOpen) {
      // Set a higher z-index when the profile dropdown is open
      setDropdownZIndex(999);
    } else {
      // Reset the z-index when the profile dropdown is closed
      setDropdownZIndex(1);
    }
  }, [profileDropdownOpen]);

  return (
    <div style={{ position: 'fixed', width: '100%', display: 'flex', justifyContent: 'space-around', zIndex: dropdownZIndex  }}>
      <div class="dropdown">
        <div class="menu-toggle" id="menu-toggle">
          <div class="line"></div>
          <div class="line"></div>
          <div class="line"></div>
          <ul class="menu-items">
            <li>
              <Link to="/home" style={linkStyle}>
                <div><h3>Home</h3></div>
              </Link>
            </li>
            <li>
              <Link to="/share-experience" style={linkStyle}>
                <div><h3>Share Experience</h3></div>
              </Link>
            </li>
            <li>
              <Link to="/search-experience" style={linkStyle}>
                <div><h3>Search Experience</h3></div>
              </Link>
            </li>
            <li>
              <Link to="/companyquestions" style={linkStyle}>
                <div><h3>Company-Wise Questions</h3></div>
              </Link>
            </li>
            <li>
              <Link to="/about-us" style={linkStyle}>
                <div><h3>About Us</h3></div>
              </Link>
            </li>
            <li>
              <Link to="/contact-us" style={linkStyle}>
                <div><h3>Contact Us</h3></div>
              </Link>
            </li>
          </ul>
        </div>
      </div>
      <div class="profile-icon" onClick={toggleProfileDropdown}>
        <img src="/profile.jpg" alt="Profile" />
        {profileDropdownOpen && (
          <ul className="profile-dropdown">
            <li>
              <Link to="/profile" style={linkStyle}>
                <div><h3>Profile</h3></div>
              </Link>
            </li>
            <li>
              <Link to="/" style={linkStyle}>
                <div style={{ color: "#d26969", marginBottom: "0%" }}><h3>Logout</h3></div>
              </Link>
            </li>
          </ul>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
