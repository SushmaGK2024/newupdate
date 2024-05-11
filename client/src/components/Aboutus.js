// components/AboutUs.js

import React, {useEffect} from 'react';
import { useNavigate } from 'react-router-dom';
const Aboutus = (props) => {
  const {currrollno}=props;
  const history = useNavigate();
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno]);
  return (
    <div>
      <h2>About Us</h2>
      {/* Display information about your platform */}
    </div>
  );
};

export default Aboutus;
