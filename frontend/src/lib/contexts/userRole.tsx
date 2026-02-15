"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

export type UserRole = "user" | "provider" | null;

interface UserRoleContextType {
  role: UserRole;
  setRole: (role: UserRole) => void;
  isLoggedIn: boolean;
  setIsLoggedIn: (loggedIn: boolean) => void;
}

export const UserRoleContext = createContext<UserRoleContextType>({
  role: null,
  setRole: () => {},
  isLoggedIn: false,
  setIsLoggedIn: () => {},
});

export const useUserRole = () => useContext(UserRoleContext);

export function UserRoleProvider({ children }: { children: ReactNode }) {
  const [role, setRole] = useState<UserRole>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  return (
    <UserRoleContext.Provider value={{ role, setRole, isLoggedIn, setIsLoggedIn }}>
      {children}
    </UserRoleContext.Provider>
  );
}
