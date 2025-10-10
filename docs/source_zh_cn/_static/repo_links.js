// 在页面加载完成后执行
window.addEventListener('DOMContentLoaded', function() {
    // 创建链接容器
    var linksContainer = document.createElement('div');
    linksContainer.className = 'project-links';

    // 创建Gitee仓库链接
    var giteeLink = document.createElement('a');
    giteeLink.href = 'https://gitee.com/aisbench/benchmark'; // Gitee仓库地址
    giteeLink.target = '_blank';
    giteeLink.className = 'project-link';
    giteeLink.innerHTML = '<i class="fa fa-github"></i> Gitee 仓库';

    // 创建官方网站链接
    var officialLink = document.createElement('a');
    officialLink.href = 'https://www.aisbench.com'; // 官方网站地址
    officialLink.target = '_blank';
    officialLink.className = 'project-link';
    officialLink.innerHTML = '<i class="fa fa-globe"></i> 官方网站';

    // 添加链接到容器
    linksContainer.appendChild(giteeLink);
    linksContainer.appendChild(officialLink);

    // 查找放置链接的位置 - 在project信息下方
    var projectNameElement = document.querySelector('.wy-side-nav-search > a');
    if (projectNameElement) {
        projectNameElement.parentNode.appendChild(linksContainer);
    } else {
        // 如果找不到特定位置，则尝试添加到页面顶部
        var headerElement = document.querySelector('.wy-side-nav-search');
        if (headerElement) {
            headerElement.appendChild(linksContainer);
        }
    }
});