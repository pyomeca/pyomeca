async function renderApiTemplate(className = "template") {
  const response = await fetch("../api/api.json");
  const json = await response.json();

  const templates = document.getElementsByClassName(className);
  for (const div of templates) {
    if (div.innerHTML) {
      setApiTemplateString(json, div);
    }
  }
}

function setApiTemplateString(json, div) {
  const linkToApi = div.innerHTML;
  const apiObject = findApiObjectByLink(json, linkToApi);
  const linkToApiP = `<p>Check out the <a href="${apiObject["link"]}">${apiObject["name"]}</a> API reference for more details.</p>`;
  div.innerHTML = apiObject["docstring"] + linkToApiP;
}

function findApiObjectByLink(obj, link) {
  if (obj.link === link) {
    return obj;
  }
  let result, p;
  for (p in obj) {
    if (obj.hasOwnProperty(p) && typeof obj[p] === "object") {
      result = findApiObjectByLink(obj[p], link);
      if (result) {
        return result;
      }
    }
  }
  return result;
}
