/*********************************************************************-                                                                                                                                             
*             7 segment digit detection with opencv                                                                                                                                                                
*                                                                                                                                                                                                                  
*           Copyright (C) 2021 By Ulrik Hørlyk Hjort                                                                                                                                                               
*                                                                                                                                                                                                                  
*  This Program is Free Software; You Can Redistribute It and/or                                                                                                                                                   
*  Modify It Under The Terms of The GNU General Public License                                                                                                                                                     
*  As Published By The Free Software Foundation; Either Version 2                                                                                                                                                  
*  of The License, or (at Your Option) Any Later Version.                                                                                                                                                          
*                                                                                                                                                                                                                  
*  This Program is Distributed in The Hope That It Will Be Useful,                                                                                                                                                 
*  But WITHOUT ANY WARRANTY; Without Even The Implied Warranty of                                                                                                                                                  
*  MERCHANTABILITY or FITNESS for A PARTICULAR PURPOSE.  See The                                                                                                                                                   
*  GNU General Public License for More Details.                                                                                                                                                                    
*                                                                                                                                                                                                                  
* You Should Have Received A Copy of The GNU General Public License                                                                                                                                                
* Along with This Program; if not, See <Http://Www.Gnu.Org/Licenses/>.                                                                                                                                             
***********************************************************************/        

#include "LedControl.h"

/* pin 12  Data in, pin 13  CLK, pin 10  LOAD  */
LedControl lc=LedControl(12,13,10,1);

void setup() {
  lc.shutdown(0,false);
  lc.setIntensity(0,0);
  lc.clearDisplay(0);
}

/*
 * Shows a random digit between 0-9 at start up.
 * Press reset to generate a new digit
 */
void loop() { 
  randomSeed(analogRead(0));
  lc.setDigit(0,2,random(10),false);
  while(1);
}
