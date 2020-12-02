#pragma once

// #include <memory>

#include "observer.h"
#include "Portals/src/objects/body.h"

// this class just links together an Observer with a Body

class Player
{
public:
	Player();
	~Player();

	void SetBodyPtr(Body* bodyPtr);
	void Synchronize(); // sets the observer data members according to the body

public:
	Observer m_Observer;
	Body* m_BodyPtr;
};









