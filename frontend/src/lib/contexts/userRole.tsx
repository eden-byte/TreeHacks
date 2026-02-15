import { createContext, useContext } from "react";

export type UserRole = "User" | "Healthcare Provider";

export const UserRoleContext = createContext<UserRole>("User");

export const useUserRole = () => useContext(UserRoleContext);
