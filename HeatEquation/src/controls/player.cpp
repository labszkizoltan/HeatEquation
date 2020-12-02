
#include "player.h"

Player::Player() : m_Observer(Observer()), m_BodyPtr(nullptr) {}

Player::~Player()
{
	m_BodyPtr = nullptr;
}

void Player::SetBodyPtr(Body * bodyPtr)
{
	m_BodyPtr = bodyPtr;
//	m_BodyPtr->angularVelocity = { 0,0,0 }; // stop the rotation of the body
}

void Player::Synchronize()
{
	m_Observer.translation = m_BodyPtr->location;
	m_Observer.orientation = m_BodyPtr->orientation;
}
